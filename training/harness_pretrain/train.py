#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "datasets>=2.18", "pyarrow>=14", "numpy"]
# ///
"""v11 Phase-1 pretrain. Runs two ways:

  standalone:  uv run train.py [config.json]        # local metrics.jsonl / ckpt/
  chuk-train:  adapted to the harness script contract (spec §5.1) -- env vars
               below take over when set, so the exact same file also runs as
               a code unit dispatched to any worker the fleet has (Colab T4,
               a rented GPU, ...).

Self-contained: vendors tiny_model_v11/ + tokenizer_v11/tokenizer.json so this
code unit carries everything it needs except the TinyStories text itself and
torch (present already on Colab; PEP 723 above resolves it standalone). Three
text sources, picked automatically at run time and never guessed at silently:

  * no `data:` block          -> stream from HuggingFace, pinned revision.
                                 Zero setup: what a viewer following along gets.
  * `data:` -> Arrow shards   -> `tiny-model/tinystories-raw`: the pinned raw
                                 text, tokenized here on the worker.
  * `data:` -> u32 stream     -> `tiny-model/v11-rust-tokenized-phase1`: already
                                 tokenized, content-addressed. No tokenizer runs
                                 at train time at all.

Tokenizer: the PUBLISHED v11 build (2026-07-24) -- crates.io `v11-core`, PyPI
`v11-tokenizer`, HF `chrishayuk/v11-tokenizer`. The vendored
`tokenizer_v11/tokenizer.json` is byte-identical to the published artifact
(sha256 10dd5110..., verified against the Hub, vocab 71260, `byte_fallback`).

NOT the native SentencePiece `v11_native.model` (sha256 4ffbfc87..., vocab
71261) that repl.py/cold_open.py use for the EXISTING checkpoint -- that is a
different id mapping and a different vocab size. Since this is a fresh pretrain
with no existing checkpoint to stay compatible with, it uses the published
build, which is also the only one of the two that is byte-safe (the native
path's mandatory metaspace step silently collapses literal multi-space runs on
decode).

That mismatch is guarded rather than trusted, because it fails *silently*: a
token stream is a flat array of integers carrying no in-band record of which
tokenizer produced it, and the catalog holds both tokenizations of TinyStories
one `curl` apart. See `check_data_identity()`: the run refuses to start unless
the staged bytes are the ones the config pinned AND decode back to plausible
English. `tokenizer_hash` in each checkpoint's meta.json is a real sha256 of the
vendored file (not a label), so a downstream consumer loading the wrong
tokenizer against this checkpoint fails loudly too.

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
import re
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

# The published v11 tokenizer (2026-07-24): crates.io v11-core 0.1.0, PyPI
# v11-tokenizer 0.1.0, HF chrishayuk/v11-tokenizer. This is the sha256 of the
# Hub's tokenizer.json, verified byte-identical to the vendored copy -- so
# "the tokenizer in this code unit is the published one" is a checked fact,
# not a claim in a comment.
PUBLISHED_TOKENIZER_SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"

# TinyStories in the chuk-datasets catalog (chuk-datasets.fly.dev). A staged
# pretrain stream is a flat array of u32 token ids and carries NO in-band
# record of which tokenizer produced it -- feed the wrong one in and training
# runs perfectly happily on nonsense. Both tokenized streams live under
# `class: pretrain-stream`, one `curl` apart. Hence check_data_identity().
CATALOG_STREAMS = {
    "67603f8ef3e67bd36676d8ad88b96f604c6d7f38ed15b7ea0910d32c93440f38": (
        "tiny-model/v11-rust-tokenized-phase1 -- tokenized with v11.vocab.bin "
        "(873f44de...), the published v11 build. CORRECT for this unit."
    ),
    "5b9d6a70601a7adaa38b594f9dba3178fc4ec4111beaf9ac5e862b01b439f7a3": (
        "tiny-model/v11-pretrain-phase1 -- tokenized with the legacy SentencePiece "
        "v11.model (4ffbfc87..., vocab 71261), the mapping the ORIGINAL v11 "
        "checkpoint was trained with. A DIFFERENT id space: WRONG for this unit."
    ),
    "41006c5696ab503e9cf99632dd497a5d414219db5fda54343205f73824e113ce": (
        "tiny-model/tinystories-raw -- raw Arrow text, pinned HF revision. "
        "Correct, and tokenized here at train time."
    ),
}

# Arrow IPC magic. HF `datasets`' on-disk cache is the *streaming* variant
# (leading continuation marker, no footer); `pa.ipc.open_file` rejects it.
ARROW_FILE_MAGIC = b"ARROW1"
ARROW_STREAM_MAGIC = b"\xff\xff\xff\xff"

# Whether a decoded sample reads like English. Measured on real TinyStories
# vs. the same text pushed through six wrong mappings (shifts of 1/7/137/1000/
# 20000, uniform-random ids, reversed ids):
#
#                       space ratio   avg word length
#   correct                  0.208            3.8
#   every wrong mapping   0.035-0.085     10.7-27.0
#
# A wrong mapping mostly yields word-piece salad with the spacing gone, so
# both signals separate by 2-3x. An earlier version of this check counted
# "plausible prose characters" instead and separated by 3% -- which would
# have passed a wrong mapping on a slightly different corpus. Thresholds sit
# between the two clusters, not next to either.
MIN_SPACE_RATIO = 0.12
MAX_AVG_WORD_LEN = 8.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


REPLAY_LOG = "train_replay.jsonl"
DONE_MARKER = "[v11-pretrain] done"


class _Tee:
    """Timestamp every stdout line into <run dir>/train_replay.jsonl.

    A 16M-token run takes ~2h and is the Act 1 centrepiece; it gets filmed by
    replaying this file, not by anyone watching two hours of terminal (see
    training/replay_run.py). Timing has to be recorded here because it cannot be
    recovered afterwards: the stalls, the checkpoint writes and the sample
    generations are most of what the wall clock consists of, and none of them
    leave a trace in metrics.jsonl.

    Costs one small append per printed line, which against a 0.4s training step
    is nothing. Never fails the run: if the log cannot be written, training
    carries on and only the replay is lost.
    """

    def __init__(self, stream, path: Path, t0: float):
        self._stream, self._path, self._t0 = stream, path, t0
        self._buf = ""
        self._ok = True

    def write(self, text):
        self._stream.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._record(line)
        return len(text)

    def _record(self, line: str):
        if not self._ok:
            return
        try:
            with self._path.open("a") as f:
                f.write(json.dumps({"t": round(time.time() - self._t0, 3),
                                    "line": line}) + "\n")
        except OSError:
            self._ok = False  # a broken replay log must not break training

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def completed_replay(run_dir: Path, total_steps: int) -> Path | None:
    """An existing log for a run that finished, at this config's step count.

    Guards against replaying something that isn't what was asked for: a run that
    died halfway, or one of a different length. Matching on total_steps catches
    the common case of editing total_tokens and expecting a fresh run.
    """
    for name in (REPLAY_LOG, "train.log"):
        path = run_dir / name
        if not path.is_file():
            continue
        text = path.read_text()
        if DONE_MARKER not in text:
            continue
        m = re.search(r"total_steps=(\d+)", text)
        if m and int(m.group(1)) == total_steps:
            return path
    return None


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


def staged_shards() -> list[Path]:
    """Every shard chuk-train staged at ./data/<sha256> (spec sections 6/7.3
    dispatch-time `data:` resolution), fetched and sha-verified once by the
    control plane rather than each worker separately streaming from
    HuggingFace. No network at train time. Sorted for determinism."""
    return sorted(p for p in Path("data").iterdir() if p.is_file())


def shard_format(path: Path) -> str:
    """`arrow-text` or `u32-stream`, sniffed from the leading bytes. The
    catalog serves both under the same content-addressed filename, so the
    name tells you nothing -- the bytes have to."""
    head = path.open("rb").read(8)
    if head.startswith(ARROW_FILE_MAGIC) or head.startswith(ARROW_STREAM_MAGIC):
        return "arrow-text"
    return "u32-stream"


def english_stats(text: str) -> tuple[float, float]:
    """(space ratio, average word length) -- the cheap, tokenizer-agnostic
    tell that a decode went through the right mapping. See the threshold
    constants for the measured separation."""
    if not text:
        return 0.0, 0.0
    words = text.split()
    avg_len = sum(len(w) for w in words) / len(words) if words else 0.0
    return sum(c.isspace() for c in text) / len(text), avg_len


def check_data_identity(shards, config, tok, vocab_size) -> None:
    """Refuse to train on bytes we cannot show belong to this tokenizer.

    A mismatched tokenizer fails *silently* -- the loss curve looks fine, the
    checkpoints save, the dashboard is green, and every number is meaningless.
    A pre-tokenized stream makes that worse, not better: there is no text to
    eyeball, just integers that are all in-range for both mappings. Two
    independent checks, both cheap:

      1. identity  -- the resolved content_sha is the one the config pinned.
      2. behaviour -- ids decode back to something that reads like English.

    (1) catches pointing the run at the wrong catalog entry. (2) catches the
    case (1) cannot: an unpinned run, or a stream whose bytes are fine but
    were produced by a tokenizer nobody recorded."""
    expected = config.get("data", {}).get("expect_content_sha", "")
    resolved = os.environ.get("CHUK_DATASET", "")
    if expected and resolved and expected != resolved:
        raise SystemExit(
            f"\n[v11-pretrain] REFUSING TO TRAIN -- dataset identity mismatch.\n"
            f"  config pinned : {expected}\n"
            f"    {CATALOG_STREAMS.get(expected, 'unknown to this unit')}\n"
            f"  harness staged: {resolved}\n"
            f"    {CATALOG_STREAMS.get(resolved, 'unknown to this unit')}\n"
            f"Fix the run's `data:` block or the config's expect_content_sha.\n"
        )

    formats = {shard_format(p) for p in shards}
    if len(formats) > 1:
        raise SystemExit(
            f"[v11-pretrain] REFUSING TO TRAIN -- mixed shard formats staged: {sorted(formats)}"
        )
    fmt = formats.pop()

    if fmt == "u32-stream":
        import numpy as np
        head = np.fromfile(shards[0], dtype="<u4", count=4096)
        if head.max() >= vocab_size:
            raise SystemExit(
                f"\n[v11-pretrain] REFUSING TO TRAIN -- staged stream contains token id "
                f"{int(head.max())}, outside this tokenizer's vocabulary (size {vocab_size}).\n"
                f"That is a different tokenizer's id space. Tokenizer loaded here: "
                f"{PUBLISHED_TOKENIZER_SHA256[:16]}... (published v11).\n"
            )
        sample = tok.decode([int(i) for i in head[:256]])
        space_ratio, avg_word_len = english_stats(sample)
        if space_ratio < MIN_SPACE_RATIO or avg_word_len > MAX_AVG_WORD_LEN:
            raise SystemExit(
                f"\n[v11-pretrain] REFUSING TO TRAIN -- staged token ids do not decode to "
                f"English through this tokenizer.\n"
                f"  space ratio     {space_ratio:.3f} (need >= {MIN_SPACE_RATIO})\n"
                f"  avg word length {avg_word_len:.1f} (need <= {MAX_AVG_WORD_LEN})\n"
                f"The ids are all in range but the mapping is wrong -- exactly the silent "
                f"failure this check exists for.\n"
                f"First 120 chars decoded: {sample[:120]!r}\n"
            )
        print(f"[v11-pretrain] data identity OK: u32-stream, {len(shards)} shard(s), "
              f"decode space_ratio={space_ratio:.3f} avg_word_len={avg_word_len:.1f}",
              flush=True)
    else:
        print(f"[v11-pretrain] data identity OK: arrow-text, {len(shards)} shard(s) "
              f"(tokenized here, at train time)", flush=True)


def _staged_texts(seed):
    """Raw text shards (`tiny-model/tinystories-raw`) -- Arrow IPC, tokenized
    on the worker. Loaded whole and shuffled in-memory (unlike the streaming
    path's windowed shuffle) since the shards are already local; deterministic
    given the same seed."""
    import random
    import pyarrow as pa
    texts = []
    for shard_path in staged_shards():
        with pa.OSFile(str(shard_path), "rb") as f:
            # HF `datasets`' own on-disk cache format is the Arrow IPC
            # *streaming* variant (sequential, no footer) -- not the file
            # variant (`pa.ipc.open_file`), which rejects these with
            # "Not an Arrow file" despite the bytes being entirely valid.
            table = pa.ipc.open_stream(f).read_all()
        texts.extend(table.column("text").to_pylist())
    random.Random(seed).shuffle(texts)
    yield from texts


def _staged_u32_batches(max_seq, batch_size):
    """Pre-tokenized streams (`tiny-model/v11-rust-tokenized-phase1`) -- flat
    little-endian u32 ids, already packed into max_seq chunks at ingest.

    Deliberately NOT reshuffled. The whole value of a content-addressed
    pre-tokenized stream is that the token order *is* the artifact: pin the
    sha and the run is bit-reproducible. Shuffling here would throw away the
    reproducibility the pin just bought, so the seed does not touch this path."""
    import numpy as np
    ids = np.concatenate([np.fromfile(p, dtype="<u4") for p in staged_shards()])
    n_chunks = len(ids) // max_seq
    chunks = ids[: n_chunks * max_seq].reshape(n_chunks, max_seq).astype(np.int64)
    for start in range(0, n_chunks - batch_size + 1, batch_size):
        yield torch.from_numpy(chunks[start : start + batch_size].copy())


def stream_batches(tok, max_seq, batch_size, seed):
    """Yields (batch_size, max_seq) long tensors -- same chunking logic as
    train_v11_replication.py's TinyStoriesDataset, just inlined so this unit
    has no dependency on tiny-model's own training module.

    Source is picked automatically, in three cases: $CHUK_DATASET set (the
    harness resolved a `data:` block and staged shards) reads them locally,
    as either pre-tokenized u32 or raw Arrow text depending on what the bytes
    actually are; unset (standalone, or a run with no `data:` block) streams
    from HuggingFace at the pinned revision."""
    if os.environ.get("CHUK_DATASET") and shard_format(staged_shards()[0]) == "u32-stream":
        yield from _staged_u32_batches(max_seq, batch_size)
        return

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
    if tokenizer_hash != PUBLISHED_TOKENIZER_SHA256:
        raise SystemExit(
            f"\n[v11-pretrain] REFUSING TO TRAIN -- vendored tokenizer is not the "
            f"published v11 build.\n  expected: {PUBLISHED_TOKENIZER_SHA256}\n"
            f"  vendored: {tokenizer_hash}\n"
            f"Re-vendor from HF chrishayuk/v11-tokenizer or v-tokenizers/v11/artifacts/.\n"
        )
    if vocab_size != arch_cfg["vocab_size"]:
        raise SystemExit(
            f"\n[v11-pretrain] REFUSING TO TRAIN -- config.json says vocab_size "
            f"{arch_cfg['vocab_size']}, tokenizer says {vocab_size}.\n"
        )

    if os.environ.get("CHUK_DATASET"):
        check_data_identity(staged_shards(), config, tok, vocab_size)

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
    total_steps_planned = max(1, total_tokens // tokens_per_step)

    # --- already trained here? then don't spend 2h reproducing it ------------
    #
    # Never on a worker: CHUK_RUN_ID is set by the control plane and by nothing
    # else, so a dispatched run always trains even if a stale log is lying about
    # in the code unit. This is purely a local convenience -- and the one that
    # makes filming Act 1e possible, since the command in the script stays the
    # same and simply replays what it already did.
    run_dir = metrics_path.parent
    on_worker = bool(os.environ.get("CHUK_RUN_ID"))
    forced = bool(os.environ.get("CHUK_FORCE_RETRAIN")) or "--force" in sys.argv
    if not on_worker and not forced:
        existing = completed_replay(run_dir, total_steps_planned)
        if existing is not None:
            print(
                f"\n[v11-pretrain] this run has already been done here, and its log is "
                f"intact.\n"
                f"  log         {existing}\n"
                f"  {total_steps_planned} steps ({total_tokens/1e6:.0f}M tokens), completed\n\n"
                f"Not retraining. The weights are already in {ckpt_dir}, and the loss\n"
                f"values in that log are the ones that produced them -- so replaying it\n"
                f"shows the actual run rather than a second one just like it:\n\n"
                f"  uv run training/replay_run.py {run_dir} --speed 60 --max-gap 2\n\n"
                f"To train again from scratch, overwriting both:\n"
                f"  CHUK_FORCE_RETRAIN=1 <same command>   (or pass --force)\n",
                flush=True)
            return

    # Record timing for replay. After the early-return above, so a replay-skip
    # never appends to the log it just declined to overwrite.
    if not on_worker:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / REPLAY_LOG).write_text("")
            sys.stdout = _Tee(sys.stdout, run_dir / REPLAY_LOG, time.time())
        except OSError:
            pass  # replay is a nicety; training is not

    resume_dir = os.environ.get("CHUK_RESUME_CKPT", "")
    start_step = 0
    if resume_dir and (Path(resume_dir) / "meta.json").is_file():
        meta = json.loads((Path(resume_dir) / "meta.json").read_text())
        start_step = int(meta.get("step", 0))
        state = load_safetensors(str(Path(resume_dir) / MODEL_FILE), device=str(device))
        model.load_state_dict(state)
        print(f"[v11-pretrain] resumed from step {start_step}", flush=True)

    total_steps = total_steps_planned
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
