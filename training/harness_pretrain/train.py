#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "datasets>=2.18", "pyarrow>=14"]
# ///
"""v11 Phase-1 pretrain. Runs two ways:

  standalone:  uv run train.py [config.json]        # local metrics.jsonl / ckpt/
  chuk-train:  adapted to the harness script contract (spec §5.1) -- env vars
               below take over when set, so the exact same file also runs as
               a code unit dispatched to any worker the fleet has (Colab T4,
               a rented GPU, ...).

Self-contained: vendors tiny_model_v11/ + tokenizer_v11/tokenizer.json so this
code unit carries everything it needs except the TinyStories text itself and
torch (present already on Colab; PEP 723 above resolves it standalone). Text
source is picked automatically at run time -- direct from HuggingFace (pinned
revision, zero setup) if no `data:` block staged anything, or the locally
staged Arrow shards chuk-datasets-server resolved if one did -- so either a
bare `uv run train.py` or a chuk-train dispatch with `data:` set just works,
without the script caring which.

Tokenizer: v-tokenizers' canonical v11 build (v11/artifacts/tokenizer.json,
loaded via the `tokenizers` library) -- NOT the native SentencePiece
`v11_native.model` tinystories-train-video's repl.py/cold_open.py use for the
existing checkpoint. That native path has a real, measured bug (SentencePiece's
mandatory metaspace step silently collapses literal multi-space runs on
decode); this build doesn't. Since this is a fresh pretrain with no existing
checkpoint to stay compatible with, there's no reason to inherit the bug.
`tokenizer_hash` in each checkpoint's meta.json is a real sha256 of the vendored
file (not a label), so a downstream consumer loading the wrong tokenizer against
this checkpoint fails loudly instead of the silent ~1e7-perplexity mismatch
Act 2c is about.

Same recipe as ~/chris-source/tiny-model/model/v11-train/train_v11_replication.py's
Phase 1 (16M tokens, seed 42) and tinystories-train-video/training/capture_emergence.py's
milestone captures (0/100k/1M/5M/16M tokens + sample generations).

Touch-points (spec §5.1): $CHUK_CONFIG (+ $CHUK_OVERRIDES), $CHUK_METRICS (JSONL:
step/loss/lr/tokens_per_s), $CHUK_CKPT_DIR (step_<n>/ dirs: model.pt + meta.json +
.ready), $CHUK_RESUME_CKPT, $CHUK_SEED/$CHUK_RUN_ID. All optional standalone --
local defaults (metrics.jsonl, ./ckpt/) apply when unset.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HUB_SHA = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # pinned, matches show_data.py
READY_MARKER = ".ready"
TOKENIZER_PATH = HERE / "tokenizer_v11" / "tokenizer.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Protocol constant (chuk-train-proto/src/constants.rs CHECKPOINT_MODEL_FILE) --
# the control plane's ingest_checkpoint() looks for exactly this filename inside
# each step_<n>/ dir and silently skips registration if it's missing. Must be
# real safetensors, not a renamed torch.save file: lazarus's load_checkpoint
# and any other downstream consumer parses it as such.
MODEL_FILE = "model.safetensors"

DEFAULT_MILESTONE_TOKENS = [0, 100_000, 1_000_000, 5_000_000, 16_000_000]


def load_config() -> dict:
    config: dict = {}
    config_path = os.environ.get("CHUK_CONFIG", "")
    # Standalone convenience: `uv run train.py configs/smoke.json` with no
    # CHUK_CONFIG set. Under the harness sys.argv is always just [script], so
    # this never fires there.
    if not config_path and len(sys.argv) > 1:
        config_path = sys.argv[1]
    if config_path and Path(config_path).is_file():
        config = json.loads(Path(config_path).read_text())
    overrides = os.environ.get("CHUK_OVERRIDES", "")
    if overrides:
        config.update(json.loads(overrides))
    return config


def _special_id(tok, token: str) -> int | None:
    return tok.token_to_id(token)


@torch.no_grad()
def generate(model, tok, prompt, device, max_seq, max_new=30, greedy=True):
    bos_id = _special_id(tok, "<s>")
    eos_id = _special_id(tok, "</s>")
    ids = tok.encode(prompt).ids
    if bos_id is not None:
        ids = [bos_id] + ids
    n_prompt = len(ids)
    for _ in range(max_new):
        window = ids[-max_seq:]
        logits = model(torch.tensor([window], device=device))[0, -1].float()
        nxt = int(logits.argmax()) if greedy else int(
            torch.multinomial(torch.softmax(logits, -1), 1))
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
    full = tok.decode(ids)
    head = tok.decode(ids[:n_prompt])
    return full[len(head):]


def _hf_stream_texts(seed):
    """Direct from HuggingFace -- the zero-setup default. What anyone
    following along at home gets with no extra infrastructure."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, revision=HUB_SHA)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    for sample in ds:
        yield sample["text"]


def _staged_texts(seed):
    """Read every shard chuk-train staged at ./data/<sha256> (spec sections
    6/7.3 dispatch-time `data:` resolution) -- plain Arrow IPC files (HF
    `datasets`' own on-disk cache format; the hash-only filename doesn't
    change the bytes), fetched once by the control plane rather than each
    worker separately streaming from HuggingFace. No network at train time.
    Loaded whole and shuffled in-memory (unlike the streaming path's
    windowed shuffle) since the shards are already local; deterministic
    given the same seed."""
    import random
    import pyarrow as pa
    texts = []
    for shard_path in sorted(Path("data").iterdir()):
        if not shard_path.is_file():
            continue
        with pa.OSFile(str(shard_path), "rb") as f:
            # HF `datasets`' own on-disk cache format is the Arrow IPC
            # *streaming* variant (sequential, no footer) -- not the file
            # variant (`pa.ipc.open_file`), which rejects these with
            # "Not an Arrow file" despite the bytes being entirely valid.
            table = pa.ipc.open_stream(f).read_all()
        texts.extend(table.column("text").to_pylist())
    random.Random(seed).shuffle(texts)
    yield from texts


def stream_batches(tok, max_seq, batch_size, seed):
    """Yields (batch_size, max_seq) long tensors -- same chunking logic as
    train_v11_replication.py's TinyStoriesDataset, just inlined so this unit
    has no dependency on tiny-model's own training module.

    Text source is picked automatically: $CHUK_DATASET set (the harness
    resolved a `data:` block and staged shards) reads them locally; unset
    (standalone, or a run with no `data:` block) streams from HuggingFace."""
    bos_id = _special_id(tok, "<s>")
    texts = _staged_texts(seed) if os.environ.get("CHUK_DATASET") else _hf_stream_texts(seed)
    buffer, batch = [], []
    for text in texts:
        ids = tok.encode(text).ids
        if bos_id is not None:
            ids = [bos_id] + ids
        buffer.extend(ids)
        while len(buffer) >= max_seq:
            batch.append(torch.tensor(buffer[:max_seq], dtype=torch.long))
            buffer = buffer[max_seq:]
            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []


def main() -> None:
    config = load_config()
    total_tokens = int(config.get("total_tokens", 16_000_000))
    milestone_tokens = sorted(config.get("milestone_tokens", DEFAULT_MILESTONE_TOKENS))
    batch_size = int(config.get("batch_size", 4))
    lr = float(config.get("lr", 3e-4))
    warmup_steps = int(config.get("warmup_steps", 100))
    metrics_every = int(config.get("metrics_every", 20))
    seed = int(os.environ.get("CHUK_SEED") or config.get("seed", 42))
    sample_prompts = config.get("sample_prompts", [
        "Once upon a time",
        "Lily had three apples. Tom gave her four more. Now Lily has",
    ])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # standalone on this Mac
    else:
        device = torch.device("cpu")
    torch.manual_seed(seed)

    arch_cfg = json.loads((HERE / "config.json").read_text())
    max_seq = arch_cfg["max_seq"]

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer_hash = sha256_file(TOKENIZER_PATH)
    vocab_size = tok.get_vocab_size()

    from tiny_model_v11 import TinyModel
    model = TinyModel(
        vocab_size=vocab_size, dim=arch_cfg["dim"], n_layers=arch_cfg["n_layers"],
        ffn_dim=arch_cfg["ffn_dim"], n_heads=arch_cfg["n_heads"], n_kv_heads=arch_cfg["n_kv_heads"],
        max_seq=arch_cfg["max_seq"],
    ).to(device)

    # Standalone falls back to local paths; the harness always sets both.
    metrics_path = Path(os.environ.get("CHUK_METRICS", "metrics.jsonl"))
    ckpt_dir = Path(os.environ.get("CHUK_CKPT_DIR", "ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = batch_size * max_seq
    milestone_steps = sorted({t // tokens_per_step for t in milestone_tokens})

    resume_dir = os.environ.get("CHUK_RESUME_CKPT", "")
    start_step = 0
    if resume_dir and (Path(resume_dir) / "meta.json").is_file():
        meta = json.loads((Path(resume_dir) / "meta.json").read_text())
        start_step = int(meta.get("step", 0))
        state = load_safetensors(str(Path(resume_dir) / MODEL_FILE), device=str(device))
        model.load_state_dict(state)
        print(f"[v11-pretrain] resumed from step {start_step}", flush=True)

    total_steps = max(1, total_tokens // tokens_per_step)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup_steps) * max(0.05, 1.0 - s / total_steps))

    def write_checkpoint(step: int):
        model.eval()
        samples = {p: generate(model, tok, p, device, max_seq) for p in sample_prompts}
        model.train()
        step_dir = ckpt_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        # embed.weight and lm_head.weight are tied (same storage, TinyModel's own
        # weight-tying) -- safetensors refuses to save aliased tensors, so clone
        # every entry to break the sharing before saving.
        state = {k: v.detach().clone().contiguous().cpu() for k, v in model.state_dict().items()}
        save_safetensors(state, str(step_dir / MODEL_FILE))
        (step_dir / "meta.json").write_text(json.dumps({
            "step": step, "arch": "tinymodel-115M dim512 L20",
            "tokenizer_hash": tokenizer_hash, "tokens": step * tokens_per_step,
            "samples": samples,
        }))
        (step_dir / READY_MARKER).touch()
        print(f"[v11-pretrain] checkpoint step_{step} ({step*tokens_per_step/1e6:.3f}M tok)", flush=True)
        for p, s in samples.items():
            print(f"  {p!r} -> {s!r}", flush=True)

    print(f"[v11-pretrain] device={device} seed={seed} total_steps={total_steps} "
          f"({total_tokens/1e6:.0f}M tokens) milestone_steps={milestone_steps}", flush=True)

    if start_step == 0 and 0 in milestone_steps:
        write_checkpoint(0)

    model.train()
    t0 = time.time()
    step = start_step
    losses = []
    with metrics_path.open("a") as mf:
        for batch in stream_batches(tok, max_seq, batch_size, seed):
            if step >= total_steps:
                break
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, vocab_size),
                batch[:, 1:].contiguous().view(-1), ignore_index=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            losses.append(loss.item())

            if step % metrics_every == 0:
                elapsed = time.time() - t0
                tok_s = (step - start_step) * tokens_per_step / max(elapsed, 1e-6)
                rec = {"step": step, "loss": round(sum(losses[-metrics_every:]) / metrics_every, 4),
                       "lr": sched.get_last_lr()[0], "tokens_per_s": round(tok_s, 1)}
                mf.write(json.dumps(rec) + "\n"); mf.flush()
                print(f"[v11-pretrain] step {step}/{total_steps} loss={rec['loss']:.4f} "
                      f"lr={rec['lr']:.2e} {tok_s:.0f} tok/s", flush=True)

            if step in milestone_steps:
                write_checkpoint(step)

    print("[v11-pretrain] done", flush=True)


if __name__ == "__main__":
    main()
