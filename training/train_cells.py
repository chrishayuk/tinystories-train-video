#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "numpy"]
# ///
"""CN-9 cells midtrain -- the DELEGATING arm, paired to train_mathonly.py.

Same base, same tokenizer, same replay fraction, same budget, same seeds as the
maths-only arm. Three things differ, and they are the experiment:

  1. VOCABULARY. Extended 71,260 -> 72,052: `<call>`, `</call>`, and one identity
     token per library cell. The maths-only arm needs no resize because its
     corpus never mentions a cell; this one is the reason the resize exists.
  2. LOSS MASKING. Every row carries a per-token mask. The result of a delegated
     call is injected with ZERO gradient, so the model is supervised on *when to
     call* and never on *what the answer is*. That is the hypothesis, and the
     mask is the only thing enforcing it.
  3. VALIDATION SPLIT is species-aware (`species == "s4"`) rather than inferred
     from row length, because this corpus has long non-replay rows -- an S3
     transcript is longer than most stories.

Everything else -- the checkpoint writer, the sampler, the validation loss, the
harness contract -- is IMPORTED from train_mathonly rather than copied. Two arms
whose shared machinery has drifted are not comparable, and a copy drifts the
moment either is touched.

    uv run train_cells.py --smoke                  # ~1 min, verifies the loop
    uv run train_cells.py --tokens 12_000_000      # the real arm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# Shared with the maths-only arm ON PURPOSE -- see the module docstring.
from train_mathonly import (  # noqa: E402
    PAD_ID, VAL_POSITION_BUDGET, report_samples, sample, val_batches,
    write_harness_ckpt,
)

ARTEFACTS = HERE.parent / "model_v11"
CORPUS = HERE / "data" / "cells" / "cells_corpus.jsonl"
TOKEN_MAP = HERE / "data" / "cells" / "cells_token_map.json"
RUN_DIR = HERE.parent / "run_cells"
DONE_MARKER = "== done:"

# The maths prompts the other arm watches, plus the two only this arm can answer.
# Kept in this order so the shared five line up with train_mathonly's output and
# the two runs can be read side by side.
CELL_PROMPTS = [
    ("The truck brought 47 crates with 63 apples in each crate. "
     "The counting machine worked it out:",
     "beyond-tier -> should EMIT A CALL, not a number (47x63=2961)"),
    ("840 sweets were shared fairly between 7 children. "
     "The sharing machine said each child gets",
     "beyond-tier division -> should emit a call (840/7=120)"),
]


def resize_embedding(model, new_vocab: int) -> int:
    """Grow the tied embedding to `new_vocab` rows, in place.

    NEW ROWS GET A SMALL NORM, AND THE CHOICE IS WORTH 0.24 NATS. Measured on
    the shared validation split, before any training:

        base, vocab 71,260                    1.5893
        + 792 rows at the table's mean/std    1.8325   (+0.2432)
        + 792 rows at std x 0.02              1.5940   (+0.0047)

    The obvious initialiser -- match the existing table's distribution -- is a
    trap here, because these embeddings are TIED. A new row is also an output
    row, so its logit is `row · hidden`; give it a real norm in a random
    direction and it wins probability mass from genuine tokens everywhere,
    including in stories that will never contain a cell. That cost is nearly
    twice the entire measured cost of the maths-only midtrain (+0.1289), so an
    arm initialised that way would start 0.24 nats in the hole and the
    comparison would be measuring the initialiser.

    Small-norm rather than exactly zero: zeros score identically but make all
    792 rows literally the same vector, and identical rows are a symmetry worth
    not relying on gradients to break.

    Tied weights: `lm_head.weight` IS `embed.weight`, so the tie is re-pointed
    after the swap. Missing that gives a model that trains an input embedding
    and reads out through a stale output matrix -- which does not error, it just
    quietly learns nothing useful for the new tokens.
    """
    old = model.embed.weight.data
    n_old, dim = old.shape
    if new_vocab <= n_old:
        return n_old
    grown = torch.empty(new_vocab, dim, dtype=old.dtype, device=old.device)
    grown[:n_old] = old
    grown[n_old:].normal_(mean=0.0, std=old.std().item() * 0.02)
    model.embed.weight = torch.nn.Parameter(grown)
    model.lm_head.weight = model.embed.weight      # re-tie, or the head is stale
    model.vocab_size = new_vocab
    return n_old


def shared_val_rows(mathonly_corpus: str):
    """The maths-only arm's held-out replay rows, exactly as it slices them.

    Duplicated from train_mathonly's main() rather than imported, because there
    it is four lines inside a 300-line function. If that slicing ever changes,
    this must change with it -- the two arms sharing a validation set is the only
    thing that makes their NLLs comparable, and a silent divergence here would
    not error, it would just quietly answer a different question.
    """
    path = Path(mathonly_corpus)
    if not path.is_file():
        raise SystemExit(
            f"\nREFUSING: need the maths-only corpus at {path} for the SHARED\n"
            f"validation split. Both arms must score the same held-out stories or\n"
            f"their replay NLLs cannot be compared, which is the point of the pair.\n"
            f"Build it with training/build_mathonly_corpus.py, or pass --val-corpus.\n")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    replay_idx = [i for i, r in enumerate(rows) if len(r["ids"]) > 40]
    return [rows[i] for i in sorted(set(replay_idx[-max(10, len(replay_idx) // 10):]))]


def masked_batch(chunk, device):
    """Pad a chunk into (ids, weight), where `weight` is the row's own loss mask.

    This is the whole difference from the maths-only batcher. There the mask is
    presence-vs-padding; here padding and a masked answer are both zero, and for
    the same reason -- neither should contribute gradient.
    """
    m = max(len(r["ids"]) for r in chunk)
    ids = torch.full((len(chunk), m), PAD_ID, dtype=torch.long)
    w = torch.zeros((len(chunk), m))
    for k, r in enumerate(chunk):
        n = len(r["ids"])
        ids[k, :n] = torch.tensor(r["ids"])
        w[k, :n] = torch.tensor(r["loss"], dtype=torch.float)
    return ids.to(device), w.to(device)


@torch.no_grad()
def val_nll_masked(model, rows, device):
    """Replay NLL, comparable to the maths-only arm's.

    Deliberately ignores the per-token mask: replay rows are unmasked anyway, and
    scoring them the same way in both arms is what makes 1.5893 -> x comparable
    across the pair. A different denominator here would silently invalidate the
    only cross-arm number the video quotes.
    """
    model.eval()
    tot, n = 0.0, 0
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
        torch.cuda.empty_cache()
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--token-map", default=str(TOKEN_MAP))
    ap.add_argument("--val-corpus", default=str(HERE / "data" / "mathonly_corpus.jsonl"),
                    help="corpus whose held-out replay rows BOTH arms validate on")
    ap.add_argument("--base-checkpoint", default="model_full.pt")
    ap.add_argument("--out", default="model_cells.pt")
    ap.add_argument("--tokens", type=int, default=12_000_000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=80)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

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
    from replay_capture import completed_run, start_capture
    if out_path.exists() and not args.force:
        done = completed_run(RUN_DIR, DONE_MARKER)
        extra = ""
        if done is not None:
            extra = (f"\n  Its log is intact, so replay it rather than retraining:\n"
                     f"    uv run training/replay_run.py {RUN_DIR.name} "
                     f"--speed 60 --max-gap 2\n")
        raise SystemExit(f"\nREFUSING to run: {out_path} already exists.\n{extra}"
                         f"\n  To train again anyway: --force.\n")
    replay_log = start_capture(RUN_DIR) if not args.smoke else None

    tmap = json.loads(Path(args.token_map).read_text())
    extended = tmap["extended_vocab"]

    from tiny_model_v11 import load_from_artifacts
    base, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint=args.base_checkpoint,
                                    device="cpu")
    n_old = resize_embedding(base, extended)
    base = base.to(device)
    print(f"== cells midtrain on {device} | base {args.base_checkpoint} | "
          f"vocab {n_old} -> {extended} (+{extended - n_old} cell tokens) ==", flush=True)

    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    # The corpus and the token map must agree, and neither carries the other's
    # identity in-band. A corpus built against a different map trains happily and
    # means something else.
    top = max(max(r["ids"]) for r in rows)
    if top >= extended:
        raise SystemExit(f"\nREFUSING: corpus holds token id {top}, beyond the "
                         f"{extended}-row table {Path(args.token_map).name} describes.\n")
    if any(len(r["ids"]) != len(r["loss"]) for r in rows):
        raise SystemExit("\nREFUSING: a row's loss mask does not match its ids.\n")

    # THE VALIDATION SET IS SHARED WITH THE MATHS-ONLY ARM, NOT DRAWN FROM THIS
    # CORPUS. Both arms replay TinyStories, but they hold out different rows --
    # measured on the same base model, this corpus's own split reads 1.8543 where
    # the maths-only arm's reads 1.5893. Neither is wrong; they are different
    # sentences. Scoring the two arms on different sentences and then quoting the
    # difference as the cost of delegation would be reporting the split.
    #
    # So both arms score the same 710 rows, and 1.5893 -> x means the same thing
    # in each. Those rows are held out of THIS corpus's training set by text, not
    # by index, because the two builds select replay independently.
    val = shared_val_rows(args.val_corpus)
    held_text = {r["text"] for r in val}
    train = [r for r in rows if r["text"] not in held_text]
    dropped = len(rows) - len(train)
    if args.smoke:
        train = train[:400]
    masked = sum(len(r["loss"]) - sum(r["loss"]) for r in train)
    tot_tok = sum(len(r["ids"]) for r in train)
    print(f"  rows: train {len(train)} | val (replay-only) {len(val)} "
          f"[shared with the maths-only arm; {dropped} overlapping rows held out]", flush=True)
    print(f"  masked: {masked:,} of {tot_tok:,} training tokens ({masked/tot_tok:.1%}) "
          f"carry no gradient -- the delegated answers", flush=True)

    nll0 = val_nll_masked(base, val, device) if val else float("nan")
    print(f"  pre-midtrain TinyStories val NLL: {nll0:.4f}", flush=True)

    from demo_common import V11Tokenizer
    tok = V11Tokenizer()

    import hashlib
    provenance = {
        "tokenizer_hash": tok.SHA256,
        "base_repo": os.environ.get("BASE_REPO") or None,
        "base_sha256": os.environ.get("EXPECT_SHA") or None,
        "corpus_identity": os.environ.get("CELLS_EXPECT_SHA") or None,
        "corpus_file_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "base_vocab": n_old,
        "extended_vocab": extended,
        "seed": args.seed,
        "run_id": os.environ.get("CHUK_RUN_ID") or None,
        "lr": args.lr,
        "batch_size": args.bs,
        "token_budget": args.tokens,
    }

    step0 = report_samples(base, tok, device, cfg.max_seq, "step 0 (base, before any cells)")
    if chuk_ckpt:
        write_harness_ckpt(base, Path(chuk_ckpt), 0, 0, cfg, provenance, step0, replay_log)

    import random
    rng = random.Random(args.seed)

    def epoch_batches():
        order = list(range(len(train)))
        rng.shuffle(order)
        W, batches = 4096, []
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
    done = False
    while not done:
        for chunk in epoch_batches():
            ids, w_full = masked_batch(chunk, device)
            lg = base(ids)[:, :-1]
            tgt, w = ids[:, 1:], w_full[:, 1:]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                 reduction="none")
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
                        "step": step,
                        "loss": round(sum(losses[-20:]) / len(losses[-20:]), 4),
                        "lr": sched.get_last_lr()[0],
                        "tokens_per_s": round(seen_tokens / max(1e-6, time.time() - t0), 1),
                    }) + "\n")
            if step % 100 == 0:
                print(f"  step {step:>6} ({seen_tokens/1e6:.2f}M tok)  "
                      f"loss {sum(losses[-100:])/len(losses[-100:]):.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if val and step % args.val_every == 0:
                print(f"  [replay check] step {step}: val NLL "
                      f"{val_nll_masked(base, val, device):.4f}", flush=True)
            if args.save_every and step % args.save_every == 0 and chuk_ckpt:
                write_harness_ckpt(base, Path(chuk_ckpt), step, seen_tokens, cfg,
                                   provenance, sample(base, tok, device, cfg.max_seq),
                                   replay_log)
            if args.sample_every and step % args.sample_every == 0:
                report_samples(base, tok, device, cfg.max_seq,
                               f"step {step} ({seen_tokens/1e6:.2f}M tok)")
                report_cells(base, tok, device, cfg.max_seq, tmap)
            if seen_tokens >= args.tokens or (args.smoke and step >= 30):
                done = True
                break

    nll1 = val_nll_masked(base, val, device) if val else float("nan")
    print(f"== done: {step} steps, {seen_tokens/1e6:.2f}M tokens | "
          f"val NLL {nll0:.4f} -> {nll1:.4f} ({time.time()-t0:.0f}s) ==", flush=True)

    final = report_samples(base, tok, device, cfg.max_seq, f"FINAL ({seen_tokens/1e6:.2f}M tok)")
    report_cells(base, tok, device, cfg.max_seq, tmap)
    if chuk_ckpt:
        write_harness_ckpt(base, Path(chuk_ckpt), step, seen_tokens, cfg, provenance,
                           final, replay_log)
    torch.save(base.state_dict(), out_path)
    print(f"saved {out_path}")


@torch.no_grad()
def report_cells(model, tok, device, max_seq, tmap, max_new=12):
    """Does it DELEGATE? The one thing the maths-only arm cannot be asked.

    Prints the raw next-token id too, because `<call>` decodes to nothing
    printable through a tokenizer that has never heard of it -- and "no visible
    output" would otherwise read as failure when it is the intended answer.
    """
    model.eval()
    call_id = tmap["call"]
    inv = {v: k for k, v in tmap["cells"].items()}
    print("  [cells] does it delegate?", flush=True)
    for prompt, note in CELL_PROMPTS:
        ids = tok.encode(prompt)
        if tok.bos_id() >= 0:
            ids = [tok.bos_id()] + ids
        emitted = []
        for _ in range(max_new):
            nxt = int(model(torch.tensor([ids[-max_seq:]], device=device))[0, -1].argmax())
            ids.append(nxt)
            emitted.append("<call>" if nxt == call_id
                           else f"⟨{inv[nxt]}⟩" if nxt in inv
                           else tok.id_to_piece(nxt))
            if nxt == tok.eos_id():
                break
        called = any(e == "<call>" for e in emitted)
        print(f"    {prompt[:58]!r}…", flush=True)
        print(f"      -> {''.join(emitted)}   {'CALLED' if called else 'no call'}", flush=True)
        print(f"      {note}", flush=True)
    model.train()


if __name__ == "__main__":
    main()
