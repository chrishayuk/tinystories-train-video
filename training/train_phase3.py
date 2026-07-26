#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "pyarrow>=14", "numpy"]
# ///
"""Phase 3 — freeze the FFN, retrain attention. Act 1f's `/compiled`, Act 4c's callback.

WHY THIS EXISTS. `model_compiled.pt` was trained once, and it is on the WRONG
LINEAGE: 71,261 embedding rows, the retired SentencePiece vocabulary, against the
published tokenizer's 71,260. It cannot be driven by this project's tokenizer at
all, and the published base's own provenance says so --
`not_included: ["phase 2 frozen-FFN attention retrain", ...]`. So phase 3 has to
be redone from `model_full.pt`, and this is the script that does it.

THE RECIPE IS THE ORIGINAL'S, NOT A NEW ONE, and it is deliberately more than
"freeze the FFN":

  1. FRESH ATTENTION. A new model is built at seed SEED+100 and the FFN, both
     norms, the embedding and the final norm are copied across from the trained
     model. Attention is left at its fresh initialisation. So this is not
     continued training -- the attention is thrown away and relearned against a
     fixed FFN.
  2. FREEZE EVERYTHING ELSE. FFN, attn_norm, ffn_norm, embed and final norm all
     get requires_grad = False. Only attention parameters move.
  3. 8M tokens at seed SEED+1, at HALF the phase-1 learning rate.

That construction is what makes Act 4c's "the same phase-two trick from act one"
mean something: stop changing what the model *knows*, and let it get better at
*routing* what it already knows.

    uv run training/train_phase3.py --smoke        # ~1 min, verifies the loop
    uv run training/train_phase3.py                # the real thing, 8M tokens
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

ARTEFACTS = HERE.parent / "model_v11"
RUN_DIR = HERE.parent / "run_phase3"
DONE_MARKER = "== done:"

# The originals, from train_v11_replication.py. Do not tune these -- the point of
# the act is that it is the same trick, at the same settings, as phase 1.
SEED, LR, BATCH_SIZE, TOKENS_PHASE3 = 42, 3e-4, 4, 8_000_000
MAX_SEQ = 256

SAMPLE_PROMPTS = [
    ("Once upon a time", "storytelling — the thing phase 3 must not break"),
    ("Tom found a lost mitten in the snow. He", "the cold open's prompt"),
    ("Lily had three apples. Tom gave her four more. Now Lily has",
     "still cannot do maths — phase 3 is not about that"),
]


def compile_ffn(trained, device, vocab_size, cfg):
    """Fresh attention, copied FFN/embed/norms — the original's construction."""
    from tiny_model_v11.model import TinyModel
    torch.manual_seed(SEED + 100)
    compiled = TinyModel(vocab_size=vocab_size, dim=cfg.dim, n_layers=cfg.n_layers,
                         n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                         ffn_dim=cfg.ffn_dim, max_seq=cfg.max_seq).to(device)
    with torch.no_grad():
        for li in range(cfg.n_layers):
            for part in ("gate", "up", "down"):
                getattr(compiled.layers[li].ffn, part).weight.data.copy_(
                    getattr(trained.layers[li].ffn, part).weight.data)
            compiled.layers[li].attn_norm.weight.data.copy_(
                trained.layers[li].attn_norm.weight.data)
            compiled.layers[li].ffn_norm.weight.data.copy_(
                trained.layers[li].ffn_norm.weight.data)
        compiled.embed.weight.data.copy_(trained.embed.weight.data)
        compiled.norm.weight.data.copy_(trained.norm.weight.data)
    return compiled


def freeze_ffn(model):
    """Only attention moves. Everything that holds knowledge is pinned."""
    for layer in model.layers:
        for p in layer.ffn.parameters():
            p.requires_grad = False
        for p in layer.attn_norm.parameters():
            p.requires_grad = False
        for p in layer.ffn_norm.parameters():
            p.requires_grad = False
    for p in model.embed.parameters():
        p.requires_grad = False
    for p in model.norm.parameters():
        p.requires_grad = False


def stream_batches(tok, budget: int, bs: int, seed: int):
    """TinyStories at the pinned revision, packed into fixed-length blocks.

    Reads the parquet shards directly over HTTP range requests, the same way
    show_data.py does and for the same reason: `datasets` cannot be imported in a
    process that also holds torch without deadlocking at shutdown.
    """
    import show_data
    import pyarrow.parquet as pq

    handle = show_data._HTTPRangeFile(show_data.DATA_URL)
    pf = pq.ParquetFile(handle)
    buf, seen = [], 0
    for group in range(pf.num_row_groups):
        rows = pf.read_row_group(group, columns=["text"]).column("text").to_pylist()
        for text in rows:
            buf.extend(tok.encode(text.strip()))
            buf.append(tok.eos_id())
            while len(buf) >= bs * (MAX_SEQ + 1):
                take = bs * (MAX_SEQ + 1)
                block = torch.tensor(buf[:take]).view(bs, MAX_SEQ + 1)
                del buf[:take]
                seen += bs * MAX_SEQ
                yield block
                if seen >= budget:
                    return
    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-checkpoint", default="model_full.pt")
    ap.add_argument("--out", default="model_compiled.pt")
    ap.add_argument("--tokens", type=int, default=TOKENS_PHASE3)
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR * 0.5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED + 1)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    chuk_metrics = os.environ.get("CHUK_METRICS", "")
    chuk_ckpt = os.environ.get("CHUK_CKPT_DIR", "")
    if os.environ.get("CHUK_SEED"):
        args.seed = int(os.environ["CHUK_SEED"])
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t0 = time.time()

    out_path = ARTEFACTS / "artifacts" / args.out
    from replay_capture import start_capture
    if out_path.exists() and not args.force:
        raise SystemExit(f"\nREFUSING to run: {out_path} already exists.\n"
                         f"  To train again anyway: --force.\n")
    replay_log = start_capture(RUN_DIR) if not args.smoke else None

    from tiny_model_v11 import load_from_artifacts
    base, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint=args.base_checkpoint,
                                    device=device)
    from demo_common import V11Tokenizer
    tok = V11Tokenizer()

    print(f"== phase 3 on {device} | base {args.base_checkpoint} | "
          f"vocab {cfg.vocab_size} ==", flush=True)
    model = compile_ffn(base, device, cfg.vocab_size, cfg)
    freeze_ffn(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    live = sum(p.numel() for p in trainable)
    print(f"  attention only: {live/1e6:.1f}M of {total/1e6:.1f}M parameters "
          f"({live/total:.1%}) carry gradient", flush=True)

    budget = 200_000 if args.smoke else args.tokens
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    seen_tokens, step, losses = 0, 0, []
    # Annealed on tokens, for the reason spelled out in train_mathonly.py: the
    # budget is denominated in tokens, so the schedule must be too.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup)
                       * max(0.05, 1.0 - seen_tokens / budget))

    provenance = {
        "tokenizer_hash": tok.SHA256, "seed": args.seed, "lr": args.lr,
        "batch_size": args.bs, "token_budget": budget, "phase": 3,
        "construction": "fresh attention @ seed 142, copied FFN/embed/norms, FFN frozen",
        "base_checkpoint": args.base_checkpoint,
        "run_id": os.environ.get("CHUK_RUN_ID") or None,
    }

    from train_mathonly import report_samples, sample, write_harness_ckpt
    step0 = report_samples(model, tok, device, cfg.max_seq, "step 0 (fresh attention)")
    if chuk_ckpt:
        write_harness_ckpt(model, Path(chuk_ckpt), 0, 0, cfg, provenance, step0, replay_log)

    model.train()
    for block in stream_batches(tok, budget, args.bs, args.seed):
        ids = block.to(device)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        seen_tokens += int(ids[:, :-1].numel())
        opt.step(); sched.step()
        losses.append(loss.item())
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
        if args.save_every and step % args.save_every == 0 and chuk_ckpt:
            write_harness_ckpt(model, Path(chuk_ckpt), step, seen_tokens, cfg,
                               provenance, sample(model, tok, device, cfg.max_seq),
                               replay_log)
        if args.sample_every and step % args.sample_every == 0:
            report_samples(model, tok, device, cfg.max_seq,
                           f"step {step} ({seen_tokens/1e6:.2f}M tok)")
        if args.smoke and step >= 30:
            break

    print(f"== done: {step} steps, {seen_tokens/1e6:.2f}M tokens "
          f"({time.time()-t0:.0f}s) ==", flush=True)
    final = report_samples(model, tok, device, cfg.max_seq, f"FINAL ({seen_tokens/1e6:.2f}M tok)")
    if chuk_ckpt:
        write_harness_ckpt(model, Path(chuk_ckpt), step, seen_tokens, cfg, provenance,
                           final, replay_log)
    torch.save(model.state_dict(), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
