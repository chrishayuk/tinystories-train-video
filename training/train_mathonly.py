#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "numpy"]
# ///
"""Self-contained maths-only (no cells) midtrain for Act 3 -- no cell80 dependency.

Loads model_v11/artifacts/model_full.pt (the same base repl.py
use), continues training on training/data/mathonly_corpus.jsonl (built by
build_mathonly_corpus.py), and saves model_v11/artifacts/model_mathonly.pt --
which repl.py's /mathonly command already knows how to load.

Because this corpus never mentions a cell, the vocabulary never needs
resizing (unlike CN-7's cell80-repo version, which always extends the
embedding for ~790 cell-identity tokens whether or not the corpus uses them)
-- this script's output is a plain, native-vocab state_dict from end to end.

Run:
  uv run train_mathonly.py --smoke                 # ~1 min, verifies the loop
  uv run train_mathonly.py --tokens 15_000_000      # real run, ~2-3h on MPS
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # vendored tiny_model_v11/ lives at the repo root

ARTEFACTS = HERE.parent / "model_v11"
CORPUS = HERE / "data" / "mathonly_corpus.jsonl"
RUN_DIR = HERE.parent / "run_mathonly"
PAD_ID = 0
DONE_MARKER = "== done:"


# Watched at every checkpoint, exactly as harness_pretrain/train.py samples the
# pretrain -- Act 3 needs its own emergence table, showing arithmetic arriving the
# way Act 1e shows fluency arriving. Each prompt answers one open question:
SAMPLE_PROMPTS = [
    ("7 + 5 =", "in-range canonical — should become correct (12)"),
    ("394 + 251 =", "one decimal digit past the range — should stay wrong (645)"),
    ("Lily had 3 apples. Tom gave Lily 4 more. Now Lily has",
     "in-range narrative, digits — the corpus surface (7)"),
    ("Lily had three apples. Tom gave her four more. Now Lily has",
     "the SAME sum in number WORDS — Act 2a's prompt, never in the corpus (seven)"),
    ("Once upon a time", "storytelling — watch for forgetting"),
]


@torch.no_grad()
def sample(model, tok, device, max_seq, max_new=14):
    """Greedy, short. Cheap enough to run at every val checkpoint."""
    model.eval()
    out = []
    for prompt, note in SAMPLE_PROMPTS:
        ids = tok.encode(prompt)
        if tok.bos_id() >= 0:
            ids = [tok.bos_id()] + ids
        n_prompt = len(ids)
        for _ in range(max_new):
            nxt = int(model(torch.tensor([ids[-max_seq:]], device=device))[0, -1].argmax())
            if nxt == tok.eos_id():
                break
            ids.append(nxt)
        full, head = tok.decode(ids), tok.decode(ids[:n_prompt])
        out.append((prompt, note, " ".join(full[len(head):].split())))
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # single-row generation, again a different shape
    return out


def report_samples(model, tok, device, max_seq, label):
    """Print the emergence row, and hand it back so a checkpoint can record it."""
    rows = sample(model, tok, device, max_seq)
    print(f"  [samples] {label}", flush=True)
    for prompt, note, cont in rows:
        print(f"    {prompt!r} -> {cont!r}", flush=True)
        print(f"        {note}", flush=True)
    return rows


def write_harness_ckpt(model, ckpt_dir: Path, step: int, tokens: int, cfg,
                       provenance: dict, samples: list | None = None):
    """step_<n>/model.safetensors + meta.json + .ready -- the layout the control
    plane's ingest looks for. `.ready` is touched last, so a checkpoint that is
    still being written is never picked up.

    Real safetensors, not a renamed torch.save: downstream consumers parse it as
    such. Tied embed/lm_head share storage, which safetensors refuses, so every
    tensor is cloned to break the aliasing first.

    meta.json carries the whole join, not just the step counter. A midtrain
    checkpoint's tokenizer, base weights and corpus are all inherited rather
    than produced here, so nothing in the bytes records them -- and a checkpoint
    paired with the wrong tokenizer generates fluent nonsense instead of
    erroring. publish_pretrain_hf.py refuses to publish without `tokenizer_hash`
    for exactly that reason, which this is what satisfies.
    """
    from safetensors.torch import save_file
    d = ckpt_dir / f"step_{step}"
    d.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().clone().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(d / "model.safetensors"))
    meta = {
        "step": step, "tokens": tokens, "arch": "tinymodel-115M dim512 L20",
        "vocab_size": cfg.vocab_size, "phase": "mathonly-midtrain",
        **provenance,
    }
    if samples is not None:
        # Act 3's emergence table as data, in the shape publish_pretrain_hf.py's
        # read_emergence() already knows how to read: prompt -> continuation.
        meta["samples"] = {prompt: cont for prompt, _note, cont in samples}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (d / ".ready").touch()
    print(f"  [harness ckpt] step_{step} ({tokens/1e6:.2f}M tok)", flush=True)


# Cap positions per validation batch, not rows. The logits tensor is
# [rows x positions x 71,260] at 4 bytes -- ~285 KB per POSITION -- so a fixed
# row count blows up on long rows: 16 replay rows x ~250 tokens is a 1.08 GiB
# allocation, which is what OOM'd a T4 (and only survived on MPS because unified
# memory hid it). 1,024 positions is ~292 MB, which any 16 GB card can take
# alongside the model and optimizer.
#
# Validation only. The training batcher is deliberately untouched: batch
# composition affects gradients, so changing it would change the experiment.
VAL_POSITION_BUDGET = 1024


def val_batches(rows, budget=VAL_POSITION_BUDGET):
    """Length-sorted, position-budgeted batches. Sorting first means a long row
    pads against other long rows instead of dragging a whole batch up to its
    length."""
    ordered = sorted(rows, key=lambda r: len(r["ids"]))
    batch: list = []
    longest = 0
    for r in ordered:
        longest_if_added = max(longest, len(r["ids"]))
        if batch and (len(batch) + 1) * longest_if_added > budget:
            yield batch
            batch, longest = [r], len(r["ids"])
        else:
            batch.append(r)
            longest = longest_if_added
    if batch:
        yield batch


def val_nll(model, rows, device, bs=16):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for chunk in val_batches(rows):
            m = max(len(r["ids"]) for r in chunk)
            ids = torch.full((len(chunk), m), PAD_ID, dtype=torch.long)
            mask = torch.zeros((len(chunk), m))
            for k, r in enumerate(chunk):
                ids[k, :len(r["ids"])] = torch.tensor(r["ids"])
                mask[k, :len(r["ids"])] = 1
            ids, mask = ids.to(device), mask.to(device)
            lg = model(ids)[:, :-1]
            tgt, w = ids[:, 1:], mask[:, 1:]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none")
            tot += float((ce * w.reshape(-1)).sum())
            n += int(w.sum())
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # val batches are a different shape to training
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    # model_full.pt, not model_compiled.pt: the Act 1e lineage is phase 1
    # only -- there is no frozen-FFN phase-3 checkpoint to mid-train from yet.
    ap.add_argument("--base-checkpoint", default="model_full.pt")
    ap.add_argument("--out", default="model_mathonly.pt")
    ap.add_argument("--tokens", type=int, default=15_000_000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=80)
    ap.add_argument("--val-every", type=int, default=2000)
    # Sampling is cheap -- 5 prompts x 14 greedy tokens -- while the replay check
    # runs 663 rows, so they should not share an interval. Emergence is the thing
    # worth high resolution: it is the Act 3 equivalent of Act 1e's table.
    ap.add_argument("--sample-every", type=int, default=250)
    # Periodic checkpoints so held-out accuracy can be read as a CURVE against
    # sightings-per-fact rather than as one number at one arbitrary budget. That
    # curve is the actual instrument for "did it learn arithmetic": taught and
    # held-out rising together means compositional structure is forming; taught
    # climbing while held-out stays flat is memorisation, visible as a widening
    # gap. With this on, the budget stops being a decision.
    ap.add_argument("--save-every", type=int, default=0,
                    help="save model_v11/artifacts/<stem>_s<step>.pt every N steps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing --out")
    args = ap.parse_args()
    # chuk-train contract (spec 5.1), all optional: unset means the local path is
    # byte-identical to before, which matters because MPS runs are in flight against
    # this file. CHUK_CKPT_DIR/CHUK_METRICS/CHUK_SEED are set only by a worker.
    import os
    chuk_metrics = os.environ.get("CHUK_METRICS", "")
    chuk_ckpt = os.environ.get("CHUK_CKPT_DIR", "")
    if os.environ.get("CHUK_SEED"):
        args.seed = int(os.environ["CHUK_SEED"])
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t0 = time.time()

    out_path = ARTEFACTS / "artifacts" / args.out
    from replay_capture import completed_run, start_capture, check_corpus_vocab
    if out_path.exists() and not args.force:
        # A result-bearing checkpoint is never overwritten. But if the run that made
        # it recorded a log, the useful answer is "here is how to watch it again"
        # rather than just "no" -- that log is the Act 3 footage.
        done = completed_run(RUN_DIR, DONE_MARKER)
        extra = ""
        if done is not None:
            extra = (f"\n  Its log is intact, so replay it rather than retraining -- "
                     f"those\n  loss values are the ones that produced this "
                     f"checkpoint:\n"
                     f"    uv run training/replay_run.py {RUN_DIR.name} "
                     f"--speed 60 --max-gap 2\n")
        raise SystemExit(
            f"\nREFUSING to run: {out_path} already exists.\n{extra}"
            f"\n  To train again anyway: --force (overwrites both the checkpoint "
            f"and the log).\n")
    if not args.smoke:
        start_capture(RUN_DIR)

    from tiny_model_v11 import load_from_artifacts
    base, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint=args.base_checkpoint, device="cpu")
    base = base.to(device)
    print(f"== maths-only midtrain on {device} | base {args.base_checkpoint} | "
          f"vocab {cfg.vocab_size} (native, no resize) ==", flush=True)

    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    # The corpus is pre-tokenized, so nothing in it says which tokenizer produced
    # it. Check before spending hours on it.
    check_corpus_vocab(rows, cfg.vocab_size, args.corpus)
    # last 10% of rows that came from TinyStories replay (identifiable: not produced by
    # drill_item's short canonical/narrative templates -- replay rows are the long ones)
    replay_idx = [i for i, r in enumerate(rows) if len(r["ids"]) > 40]
    val_set = set(replay_idx[-max(10, len(replay_idx) // 10):])
    val = [rows[i] for i in sorted(val_set)]
    train = [r for i, r in enumerate(rows) if i not in val_set]
    if args.smoke:
        train = train[:400]
    print(f"  rows: train {len(train)} | val (replay-only) {len(val)}", flush=True)

    # The denominator that matters is sightings per FACT, not tokens: CN-7's
    # headline (0.90 in-tier) is denominated that way, at ~1.8/epoch. Printed here
    # so the regime is on the record before anything is trained in it.
    import re as _re
    _seen = {}
    for _r in train:
        for _rx in (r"(\d+) \+ (\d+)", r"(\d+) plus (\d+)"):
            for _m in _re.finditer(_rx, _r["text"]):
                _k = (int(_m.group(1)), int(_m.group(2)))
                _seen[_k] = _seen.get(_k, 0) + 1
    if _seen:
        _corpus_tok = sum(len(r["ids"]) for r in train)
        _ep = args.tokens / max(1, _corpus_tok)
        _per = sum(_seen.values()) / len(_seen)
        print(f"  add: {sum(_seen.values()):,} items over {len(_seen):,} ordered pairs "
              f"= {_per:.2f}/epoch x {_ep:.2f} epochs = {_per*_ep:.1f} sightings/pair "
              f"(CN-7: ~1.8/epoch, ~3.6 total)", flush=True)

    nll0 = val_nll(base, val, device) if val else float("nan")
    print(f"  pre-midtrain TinyStories val NLL: {nll0:.4f}", flush=True)

    from demo_common import V11Tokenizer
    tok = V11Tokenizer()

    # What this checkpoint inherited rather than produced. None of it is
    # recoverable from the weights, and every item was verified by something
    # upstream rather than asserted here: V11Tokenizer refuses to construct
    # unless the file on disk hashes to the published build, and the entrypoint
    # refuses to start unless the base checkpoint and the corpus hash to the
    # identities named in these environment variables.
    import hashlib
    provenance = {
        "tokenizer_hash": tok.SHA256,
        "base_repo": os.environ.get("BASE_REPO") or None,
        "base_sha256": os.environ.get("EXPECT_SHA") or None,
        "corpus_identity": os.environ.get("MATHONLY_EXPECT_SHA") or None,
        "corpus_file_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "seed": args.seed,
        "run_id": os.environ.get("CHUK_RUN_ID") or None,
        "lr": args.lr,
        "batch_size": args.bs,
        "token_budget": args.tokens,
    }

    step0 = report_samples(base, tok, device, cfg.max_seq, "step 0 (base, before any maths)")

    # Prove the checkpoint WRITE path before producing anything worth losing.
    # A run that trains for forty minutes and then discovers its first save
    # boundary cannot upload has spent the forty minutes; step_0 costs seconds
    # and one 460MB write, and it is also the honest zero row of the emergence
    # curve -- the base model's answers, from this run, on this device.
    #
    # It makes the failure VISIBLE early, not fatal: the trainer has no
    # credentials to ask the control plane whether the bytes landed. Checking
    # `list_checkpoints` a minute in is the other half, and belongs to whoever
    # dispatched the run.
    if chuk_ckpt:
        write_harness_ckpt(base, Path(chuk_ckpt), 0, 0, cfg, provenance, step0)

    import random
    rng = random.Random(args.seed)

    def epoch_batches():
        order = list(range(len(train)))
        rng.shuffle(order)
        W = 4096
        batches = []
        for w in range(0, len(order), W):
            win = sorted(order[w:w + W], key=lambda i: len(train[i]["ids"]))
            for i in range(0, len(win), args.bs):
                batches.append([train[j] for j in win[i:i + args.bs]])
        rng.shuffle(batches)
        return batches

    trainable = list(base.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps_est = max(1, args.tokens // (args.bs * 30))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.05, 1.0 - s / total_steps_est))

    base.train()
    seen_tokens, step, losses = 0, 0, []
    log = []
    done = False
    while not done:
        for chunk in epoch_batches():
            m = max(len(r["ids"]) for r in chunk)
            ids = torch.full((len(chunk), m), PAD_ID, dtype=torch.long)
            mask = torch.zeros((len(chunk), m))
            for k, r in enumerate(chunk):
                ids[k, :len(r["ids"])] = torch.tensor(r["ids"])
                mask[k, :len(r["ids"])] = 1
            ids, mask = ids.to(device), mask.to(device)
            lg = base(ids)[:, :-1]
            tgt, w = ids[:, 1:], mask[:, 1:]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none")
            loss = (ce * w.reshape(-1)).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
            seen_tokens += int(ids.numel())
            step += 1
            if chuk_metrics and step % 20 == 0:
                with open(chuk_metrics, "a") as mf:
                    mf.write(json.dumps({
                        "step": step, "loss": round(sum(losses[-20:]) / len(losses[-20:]), 4),
                        "lr": sched.get_last_lr()[0],
                        "tokens_per_s": round(seen_tokens / max(1e-6, time.time() - t0), 1),
                    }) + "\n")
            if step % 100 == 0:
                print(f"  step {step:>6} ({seen_tokens/1e6:.2f}M tok)  "
                      f"loss {sum(losses[-100:])/len(losses[-100:]):.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if val and step % args.val_every == 0:
                nv = val_nll(base, val, device)
                log.append({"step": step, "tokens": seen_tokens, "val_nll": nv})
                print(f"  [replay check] step {step}: val NLL {nv:.4f}", flush=True)
            if args.save_every and step % args.save_every == 0:
                # ONE checkpoint per boundary, not two. On a worker the local .pt
                # is pure redundancy -- the control plane ingests CHUK_CKPT_DIR,
                # uploads it with lineage, and applies its own retention
                # (keep_last/keep_every from the run spec), so writing both meant
                # 1.2GB of identical weights per boundary into the job's /tmp
                # working dir with nothing pruning the local half.
                #
                # Observed 2026-07-25 on a Colab T4: the run died silently at the
                # first save boundary and the worker restarted it from the
                # entrypoint, nine times, never getting past step ~1600. No
                # traceback, which is what a SIGKILL looks like.
                if chuk_ckpt:
                    # Sampled here rather than reusing the --sample-every report:
                    # the two intervals need not coincide, and a milestone whose
                    # recorded generations came from a different step is worse
                    # than one with none. Greedy, so this is a repeat of work
                    # done moments later at best, not a divergence.
                    write_harness_ckpt(base, Path(chuk_ckpt), step, seen_tokens, cfg,
                                       provenance, sample(base, tok, device, cfg.max_seq))
                else:
                    snap = out_path.with_name(f"{out_path.stem}_s{step}{out_path.suffix}")
                    torch.save(base.state_dict(), snap)
                    print(f"  [saved] {snap.name} ({seen_tokens/1e6:.2f}M tok)", flush=True)
            if args.sample_every and step % args.sample_every == 0:
                report_samples(base, tok, device, cfg.max_seq,
                               f"step {step} ({seen_tokens/1e6:.2f}M tok)")
            if seen_tokens >= args.tokens or (args.smoke and step >= 30):
                done = True
                break

    nll1 = val_nll(base, val, device) if val else float("nan")
    print(f"== done: {step} steps, {seen_tokens/1e6:.2f}M tokens | "
          f"val NLL {nll0:.4f} -> {nll1:.4f} ({time.time()-t0:.0f}s) ==", flush=True)

    final = report_samples(base, tok, device, cfg.max_seq, f"FINAL ({seen_tokens/1e6:.2f}M tok)")

    if chuk_ckpt:
        write_harness_ckpt(base, Path(chuk_ckpt), step, seen_tokens, cfg, provenance, final)

    torch.save(base.state_dict(), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
