# tinystories-train-video

Working folder for a video on training a tiny language model from scratch and
then teaching it maths three different ways.

**Arc:** pre-train TinyStories → ask it maths, it fails → mid-train maths (no
cells) → mid-train cell calls → why answer-only training can't work.

This is the code and assets behind it — training scripts, demo tools, and the
tokenizer/checkpoint config. All public.

## Layout

```
repl.py                ★ interactive — type prompts live on camera
cold_open.py           the same demos pre-canned, for rehearsal and reference
demo_tokenizer.py      the on-camera tokenizer demos
show_data.py           what TinyStories actually looks like (Act 1c)
run_broker.sh          ★ the closing shot — model calls a Z80 cell, narrates the answer

tiny_model_v11/        vendored model code (3 files, from tiny-model/model/v11-core)

model_v11/               where the Act 1e checkpoint will land (empty — see below)

tokenizer/
  tokenizer.json              ★ the PUBLISHED v11 build — the only tokenizer here
  v11_tokenizer_README.md     the knowledge-first tokenizer description

unit.toml                the chuk-train code-unit manifest (must live at repo root)
configs/
  real.json                   ★ Act 1e — the GPU run, pinned to the pre-tokenized stream
  real_raw.json               same run from raw text, tokenizing on the worker
  smoke.json                  20k tokens, no infra — HF streaming fallback

training/
  build_mathonly_corpus.py    ★ Act 3 corpus — self-contained, no cell80 dependency
  train_mathonly.py           ★ Act 3 midtrain — loads model_compiled.pt, saves model_mathonly.pt
  cliff_probe_mathonly.py     ★ Act 3c cliff table — in-range vs. one/two-digit-past
  harness_pretrain/           ★ Act 1e — the pretrain code unit dispatched to a GPU
    train.py                    the run itself; guards, then trains
    config.json                 ★ the only architecture config (vocab 71260)
    tokenizer_v11/              the published tokenizer.json, byte-identical to the Hub's
    run.sh                      entrypoint (deliberately does not touch torch)

source-docs/
  CN7-prereg.md          background/citation only — CN-7's own (cell80) mixed-run pre-registration
  CN7-findings.md        background/citation only — results; the CN-7 section starts at "## CN-7 R1"
```

## Running anything here

Every script declares its own dependencies inline (PEP 723), so each is just:

```bash
uv run repl.py
uv run cold_open.py --slots
uv run demo_tokenizer.py --section 2
uv run show_data.py --tokens
```

No install step, no venv, no `--with` flags, and **no warnings on screen** —
`numpy` is declared so torch doesn't complain about it. The first run of each
prints one `Installed N packages` line while uv resolves; every run after is
silent. **Warm the cache by running each script once before you film.**

## repl.py — the on-camera tool

Type a prompt, press enter, watch it generate **token by token**. Nothing
pre-canned — every word on screen is produced live, which is the whole point for
video.

```
/slow            ≈15 tok/s — readable on camera (default is ~75, too fast)
/greedy          most likely token every time; no randomness to blame
/sample /temp 0.8
/next <prompt>   top-10 next words as a bar chart, instead of generating
/len 60          how many tokens to generate
/full /compiled  switch checkpoint: after phase 1/2 vs after phase 3
/mathonly        switch checkpoint: maths mid-trained, no cells (Act 3; once trained)
/fast /help /quit
```

Ctrl-C stops a generation without leaving the REPL — useful when a story rambles.

**Filming order for the cold open:** `/slow`, type *Once upon a time*, let it
stream. Then `/greedy` and type *Lily had three apples. Tom gave her four more.
Now Lily has*. Then `/next` on the four number slots below.

`cold_open.py` runs the same three demos non-interactively (`--story`, `--maths`,
`--slots`) — handy for rehearsing and for checking output before filming.

### What `--slots` shows, and why it matters

It's tempting to assume the model "puts a number in, and the number is wrong."
**That is not what happens** — it doesn't produce a number at all. It narrates
straight past the slot.

Looking at the probabilities explains why (⚠️ **these numbers came from the
retired checkpoint** and must be re-measured on the Act 1e model before they go
on screen — the shape of the result is what matters, not the exact figures):

| slot | number-word mass |
|---|---|
| `Once upon a time there were ___` | **99.1%** (`two` at 0.989) |
| `She counted the apples. There were ___` | 2.5% |
| `Lily had three apples… Now Lily has ___` | 1.2% |
| `Tom had two cats and one dog. Altogether he had ___` | 3.4% |

It is 98.9% certain about "two" where a story convention demands it, and reaches
for "a"/"some"/"many" wherever arithmetic does. Number words are narrative
texture to this model, not quantities — a ~40× swing between idiom and sum.

## Running the tokenizer demos

No repo paths, no network — reads only from `./tokenizer/`.

```bash
uv run demo_tokenizer.py
uv run demo_tokenizer.py --section 2
```

| Section | Script beat | Shows |
|---|---|---|
| 1 | Act 1b | what a token is — `Once upon a time` → 4 pieces |
| 2 | Act 2b | the vocabulary walked in ID order; the hand-built blocks |
| 3 | Act 2b | number words are 1 token, digits split one per token |
| 4 | Act 1a | 36.5M of 115.1M parameters is just the embedding table |


## The published v11 tokenizer

Published 2026-07-24. This is **the** tokenizer — everything new here uses it
and nothing else, and everything in Act 2 is independently checkable:

```bash
pip install v11-tokenizer     # PyPI 0.1.0
cargo add v11-core            # crates.io 0.1.0 (also v11-builder, v11-cli)
# huggingface.co/chrishayuk/v11-tokenizer
```

sha256 `10dd5110…`, vocab **71260**, byte-safe. Vendored into the Act 1e code
unit at `training/harness_pretrain/tokenizer_v11/tokenizer.json`, verified
byte-identical to the Hub copy. The catalog's Rust-tokenized TinyStories streams
were built from the same vocabulary (`v11.vocab.bin`, `873f44de…`).

Every script here loads it, and each verifies the file's sha256 on startup —
so an ID that goes on camera is one a viewer can reproduce from PyPI.

### The old tokenizer is gone

`v11_native.model` and `tokenizer_committed.json` have been deleted. One
tokenizer, one vocabulary, one model lineage.

The consequence is real and worth stating plainly: the **pre-existing**
checkpoint was trained on the old SentencePiece build (vocab 71261) and cannot
be driven by the published tokenizer — the ids mean different things. So
`repl.py` and `cold_open.py` now refuse against it rather than generating
fluent nonsense:

```
  checkpoint/tokenizer mismatch -- refusing to generate.
    model_compiled.pt: vocab 71,261
    published v11 tokenizer: vocab 71,260
```

**Every demo that generates text is therefore blocked on the Act 1e pretrain.**
`demo_tokenizer.py` and `show_data.py` need no model and work today. See
SCRIPT.md "What still needs running" items 1 and 5 for the sequencing — Act 4
is the long pole, because its cell80-side checkpoints need rebuilding too.

## run_broker.sh — the closing shot

```bash
./run_broker.sh
```

```
PROMPT: 157 sweets were shared fairly between 16 children.
        The sharing machine said each child gets
  [broker] model called safe_div(157, 16) -> cell returned 9
OUTPUT: <call> ⟨safe_div⟩ 157 16 </call> 9 sweets. The children smiled.
```

⚠️ **Blocked.** Verified working 2026-07-23 against the *retired* checkpoint. Its cell80-side checkpoint was built on the old vocabulary and has to be rebuilt before this runs again — see SCRIPT.md item 5c.

Unlike the other scripts this one reaches into the cell80 repo, because that's
where the mid-trained checkpoint and the Z80 executor live. It wraps two traps
that cost real time to rediscover, both documented in its comments:

- **Most `cell80_py` builds in the uv cache are too old** and reject
  `safe_div.rs` with "unsupported statement expression". There's no installable
  pin — the script hardcodes a build that works, plus a search loop to find
  another if it's ever evicted.
- **The default `--max-tokens 60` loops** the call and narration three times,
  which reads as a bug on camera. Pinned to 21, which lands exactly on "The
  children smiled."

## Other on-camera traps

- transformers needs `attn_implementation="eager"` under torch 2.6.0 on this Mac,
  or it aborts on MPS.
- The pinned `cell80_py` extension lives only in the uv cache and needs the
  PYTHONPATH trick.

## Still to run

Status as of 2026-07-25: the maths-only midtrain (`training/build_mathonly_corpus.py`
→ `train_mathonly.py` → `cliff_probe_mathonly.py`) is self-contained and
smoke-tested, no cell80 dependency — `train_mathonly.py` loads
`model_v11/artifacts/model_compiled.pt` and saves straight to
`model_mathonly.pt` (no vocab resize needed, this corpus never mentions a
cell). `repl.py`'s `/mathonly` command already loads the result directly. None
of the real training runs — Act 1e's pretrain or Act 3's midtrain — have been
fired yet.

`training/harness_pretrain/` is a from-scratch pretrain unit adapted to the
[chuk-train](https://github.com/chrishayuk/gpu-training-harness) script
contract, for running the pretrain on a GPU worker (Colab/rented) instead of
this Mac, with a live metrics dashboard. **This is Act 1e** — see below.

## Act 1e: the pretrain on real hardware

Three moving parts, all live:

| Part | Where |
|---|---|
| Tokenizer | published v11 (`10dd5110…`), vendored in the code unit |
| Data | `chuk-datasets.fly.dev` — `tiny-model/v11-rust-tokenized-phase1 @ 67603f8e…` |
| Compute | `chuk-mcp-training.fly.dev` — `submit_run`, then the dashboard |

```console
# code = this repo, entrypoint = train, config = configs/real.json
# configs/real_raw.json is the same run from raw text instead of the
# pre-tokenized stream (tokenizes on the worker; slower, not bit-reproducible)
```

Run it locally with no infrastructure at all — it falls back to streaming
TinyStories from HuggingFace at the pinned revision:

```bash
uv run training/harness_pretrain/train.py configs/smoke.json
```

### It refuses rather than trains on nonsense

The catalog holds **two** tokenized TinyStories, one `curl` apart, under the
same `pretrain-stream` class — and a token stream is a flat array of integers
with no in-band record of which tokenizer made it. Point a run at the wrong one
and it would train perfectly happily and hand you a loss curve.

So `train.py` won't start unless it can show the bytes belong to its tokenizer.
Four refusals, all tested:

| Check | Catches |
|---|---|
| vendored tokenizer sha ≠ published | a tampered/stale code unit |
| staged `content_sha` ≠ config's pin | pointing at the wrong catalog entry |
| any token id ≥ vocab_size | a different, larger id space (e.g. the 71261 build) |
| decoded sample isn't English | the case the pin can't catch — wrong mapping, right size |

That last one is deliberately blunt: it decodes the first 256 ids and checks the
space ratio and average word length. Measured separation across six wrong
mappings — correct text is 0.208 spaces / 3.8 chars-per-word, every wrong
mapping lands at 0.035–0.085 / 10.7–27.0. Thresholds sit between the clusters
(0.12 and 8.0), not next to either. An earlier version of this check counted
"plausible prose characters" and separated by 3%; it was replaced, because a 3%
margin would have passed a wrong mapping on a slightly different corpus.

**Status (2026-07-25).** Smoke-tested end to end on MPS against a staged token
stream: guard passed, 29 steps, three milestone checkpoints with real
`model.safetensors` + `meta.json` + `.ready`, samples generated, metrics
streamed, loss 10.9 → 8.0. All four refusal paths verified. **The real 16M-token
dispatch has not been fired yet** — and it needs a read-scoped chuk-datasets key,
since only `/v1/datasets` is open without auth.
