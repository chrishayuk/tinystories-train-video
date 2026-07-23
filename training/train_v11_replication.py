#!/usr/bin/env python3
"""v11 replication under the pinned-revision harness.

Why this exists: v11's original training (train_tinystories.py) never
pinned a TinyStories hub revision, so its exact training-time document set
can't be reconstructed from today's hub state -- eval_held_out.py's
contamination check can verify the three v12 candidates are clean against
the frozen held-out set, but for v11 it can only ASSUME cleanliness. This
script replicates v11's ORIGINAL training methodology exactly (same
architecture, same native v11.model tokenizer, same fixed 16M/8M token
budget, same seeds, same two-phase full+frozen-FFN recipe) with one
change: `revision=HUB_SHA` pinned on both dataset loads, matching every
other pinned run in this funnel. That makes the exact set of documents
consumed fully reproducible, so eval_held_out.py can hash-check it against
the held-out set the same way it already does for the v12 candidates.

This intentionally does NOT fix v11's fixed-token-budget (vs byte-matched)
methodology -- that's a separate, disclosed C9 gap in v11's original
design. The point here is only to verify contamination status of the
ORIGINAL methodology under a pinned, reproducible revision, not to
redesign v11's training.

Writes into `<repo>/model/v11/artifacts_pinned_replication/` (NOT the
original artifacts/ dir -- v11's real, already-referenced checkpoints stay
untouched):
    model_full.pt        -- after Phase 2 (16M tokens, seed=42)
    model_compiled.pt    -- after Phase 3 (8M tokens, seed=43, frozen FFN)
    training_results.json

Usage (from tiny-model repo root):
  uv run --project model/v11-train python model/v11-train/train_v11_replication.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from tiny_model_v11 import TinyModel, load_config
from train_tok2 import CHRIS_EXPERIMENTS_V11_MODEL, HUB_SHA, NativeSPTokenizer

REPO = Path(__file__).resolve().parents[2]  # tiny-model/
V11_ARTIFACT_DIR = REPO / "model" / "v11"
OUTPUT_DIR = V11_ARTIFACT_DIR / "artifacts_pinned_replication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = load_config(V11_ARTIFACT_DIR)
BATCH_SIZE = 4
LR = 3e-4
SEED = 42
TOKENS_PHASE1 = 16_000_000
TOKENS_PHASE3 = 8_000_000


class TinyStoriesDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq, max_tokens, seed):
        self.tok = tokenizer
        self.max_seq = max_seq
        self.max_tokens = max_tokens
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, revision=HUB_SHA)
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)
        tokens_seen = 0
        buffer = []
        for sample in ds:
            ids = self.tok.encode(sample["text"], add_special_tokens=True,
                                    max_length=self.max_seq * 2, truncation=True)
            buffer.extend(ids)
            while len(buffer) >= self.max_seq:
                chunk = buffer[:self.max_seq]
                buffer = buffer[self.max_seq:]
                tokens_seen += len(chunk)
                if tokens_seen > self.max_tokens:
                    return
                yield torch.tensor(chunk, dtype=torch.long)


def collate_fn(batch):
    return torch.stack(batch)


def make_model(device, vocab_size):
    return TinyModel(
        vocab_size=vocab_size,
        dim=CONFIG.dim,
        n_layers=CONFIG.n_layers,
        ffn_dim=CONFIG.ffn_dim,
        n_heads=CONFIG.n_heads,
        n_kv_heads=CONFIG.n_kv_heads,
        max_seq=CONFIG.max_seq,
    ).to(device)


def train(model, loader, device, max_tokens, vocab_size, lr, label):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Training: {label}")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable / total:.1%})")
    print(f"  LR: {lr}")

    model.train()
    t0 = time.time()
    tokens_done = 0
    epoch_loss = 0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, vocab_size),
            batch[:, 1:].contiguous().view(-1),
            ignore_index=0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
        tokens_done += batch.numel()
        if n_batches % 200 == 0:
            avg = epoch_loss / n_batches
            elapsed = time.time() - t0
            tps = tokens_done / elapsed
            eta = (max_tokens - tokens_done) / tps if tps > 0 else 0
            print(f"    {tokens_done / 1e6:.1f}M/{max_tokens / 1e6:.0f}M  "
                  f"loss={avg:.4f}  {tps:.0f} tok/s  ETA={eta / 60:.0f}min")
            sys.stdout.flush()
        if tokens_done >= max_tokens:
            break
    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"  Done: loss={avg_loss:.4f}, {time.time() - t0:.0f}s, "
          f"{tokens_done / 1e6:.1f}M tokens")
    return avg_loss


def compile_ffn(trained_model, device, vocab_size):
    print("\n  Compiling FFN (fresh attention + copied FFN/embed/norms)…")
    torch.manual_seed(SEED + 100)
    compiled = make_model(device, vocab_size)
    with torch.no_grad():
        for li in range(CONFIG.n_layers):
            compiled.layers[li].ffn.gate.weight.data.copy_(
                trained_model.layers[li].ffn.gate.weight.data)
            compiled.layers[li].ffn.up.weight.data.copy_(
                trained_model.layers[li].ffn.up.weight.data)
            compiled.layers[li].ffn.down.weight.data.copy_(
                trained_model.layers[li].ffn.down.weight.data)
            compiled.layers[li].attn_norm.weight.data.copy_(
                trained_model.layers[li].attn_norm.weight.data)
            compiled.layers[li].ffn_norm.weight.data.copy_(
                trained_model.layers[li].ffn_norm.weight.data)
        compiled.embed.weight.data.copy_(trained_model.embed.weight.data)
        compiled.norm.weight.data.copy_(trained_model.norm.weight.data)
    return compiled


def freeze_ffn(model):
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


def main():
    print("=" * 70)
    print("  v11 REPLICATION: pinned-revision harness (hub_sha=%s)" % HUB_SHA[:12])
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Config: dim={CONFIG.dim} L={CONFIG.n_layers} "
          f"ffn={CONFIG.ffn_dim} heads={CONFIG.n_heads} kv={CONFIG.n_kv_heads}")

    print(f"\n[tokenizer] loading real v11 SP from {CHRIS_EXPERIMENTS_V11_MODEL}")
    tok = NativeSPTokenizer(CHRIS_EXPERIMENTS_V11_MODEL)
    print(f"  vocab_size = {tok.vocab_size}")

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: full training ({TOKENS_PHASE1 / 1e6:.0f}M tokens, seed={SEED})")
    print(f"{'=' * 70}")
    torch.manual_seed(SEED)
    model = make_model(device, tok.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    dataset1 = TinyStoriesDataset(tok, CONFIG.max_seq, TOKENS_PHASE1, seed=SEED)
    loader1 = DataLoader(dataset1, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    loss1 = train(model, loader1, device, TOKENS_PHASE1, tok.vocab_size, LR, label="Full v11 (pinned replication)")

    torch.save(model.state_dict(), OUTPUT_DIR / "model_full.pt")
    print(f"  saved {OUTPUT_DIR / 'model_full.pt'}")

    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: frozen FFN + retrain attention "
          f"({TOKENS_PHASE3 / 1e6:.0f}M tokens, seed={SEED + 1} @ lr={LR * 0.5})")
    print(f"{'=' * 70}")
    compiled = compile_ffn(model, device, tok.vocab_size)
    freeze_ffn(compiled)
    dataset3 = TinyStoriesDataset(tok, CONFIG.max_seq, TOKENS_PHASE3, seed=SEED + 1)
    loader3 = DataLoader(dataset3, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    loss3 = train(compiled, loader3, device, TOKENS_PHASE3, tok.vocab_size,
                   LR * 0.5, label="Attention-only on frozen FFN (pinned replication)")

    torch.save(compiled.state_dict(), OUTPUT_DIR / "model_compiled.pt")
    print(f"  saved {OUTPUT_DIR / 'model_compiled.pt'}")

    results = {
        "loss_full": loss1,
        "loss_compiled": loss3,
        "tokenizer": str(CHRIS_EXPERIMENTS_V11_MODEL),
        "vocab_size": tok.vocab_size,
        "n_params": n_params,
        "tokens_phase1": TOKENS_PHASE1,
        "tokens_phase3": TOKENS_PHASE3,
        "hub_sha": HUB_SHA,
        "seed_phase1": SEED,
        "seed_phase3": SEED + 1,
        "arch": CONFIG.to_dict(),
        "note": "pinned-revision replication of v11's original training methodology "
                "(train_tinystories.py) -- same arch/tokenizer/budget/seeds, only "
                "revision=HUB_SHA added, so training-consumed documents are fully "
                "reproducible and hash-checkable against the frozen held-out set.",
    }
    with open(OUTPUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    print(f"  Phase 2 full:       loss={loss1:.4f}")
    print(f"  Phase 3 frozen FFN: loss={loss3:.4f}")
    print(f"  Δ (improvement):    {loss1 - loss3:+.4f}")


if __name__ == "__main__":
    main()
