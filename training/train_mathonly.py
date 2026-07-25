#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "numpy"]
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


def val_nll(model, rows, device, bs=16):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
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
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing --out")
    args = ap.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
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

    nll0 = val_nll(base, val, device) if val else float("nan")
    print(f"  pre-midtrain TinyStories val NLL: {nll0:.4f}", flush=True)

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
            if step % 100 == 0:
                print(f"  step {step:>6} ({seen_tokens/1e6:.2f}M tok)  "
                      f"loss {sum(losses[-100:])/len(losses[-100:]):.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if val and step % args.val_every == 0:
                nv = val_nll(base, val, device)
                log.append({"step": step, "tokens": seen_tokens, "val_nll": nv})
                print(f"  [replay check] step {step}: val NLL {nv:.4f}", flush=True)
            if seen_tokens >= args.tokens or (args.smoke and step >= 30):
                done = True
                break

    nll1 = val_nll(base, val, device) if val else float("nan")
    print(f"== done: {step} steps, {seen_tokens/1e6:.2f}M tokens | "
          f"val NLL {nll0:.4f} -> {nll1:.4f} ({time.time()-t0:.0f}s) ==", flush=True)

    torch.save(base.state_dict(), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
