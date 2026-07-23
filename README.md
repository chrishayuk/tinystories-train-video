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

model_v11/
  config.json                 architecture (dim 512, 20 layers, 115.1M params)
  artifacts -> …              symlink to the real checkpoints (460MB each)

tokenizer/
  v11_native.model            SentencePiece model v11 was ACTUALLY trained with
  tokenizer_committed.json    the one committed next to the checkpoint (wrong IDs)
  v11_tokenizer_README.md     the knowledge-first tokenizer description

training/
  train_v11_replication.py    the pre-training recipe shown in Act 1 (on-screen only, not run from here)
  v11_config.json             same config, kept beside the recipe
  v11_training_results.json   the original run's numbers
  build_mathonly_corpus.py    ★ Act 3 corpus — self-contained, no cell80 dependency
  train_mathonly.py           ★ Act 3 midtrain — loads model_compiled.pt, saves model_mathonly.pt
  cliff_probe_mathonly.py     ★ Act 3c cliff table — in-range vs. one/two-digit-past

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

Looking at the probabilities explains why:

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
uv run demo_tokenizer.py --section 5
```

| Section | Script beat | Shows |
|---|---|---|
| 1 | Act 1b | what a token is — `Once upon a time` → 4 pieces |
| 2 | Act 2b | the vocabulary walked in ID order; the hand-built blocks |
| 3 | Act 2b | number words are 1 token, digits split one per token |
| 4 | Act 1a | 36.5M of 115.1M parameters is just the embedding table |
| 5 | Act 2c | the two mappings: identical pieces, different IDs |

## The two tokenizers — read this before filming

`v11_native.model` is what the checkpoint was trained against.
`tokenizer_committed.json` is what sits beside the checkpoint in the repo, and it
is **a different piece→ID mapping**.

It produces byte-identical *pieces* and completely different *IDs*. Nothing looks
wrong until you feed the IDs to the model, at which point it scores ~18 nats on
plain English — worse than uniform random guessing (~11). Through the native
mapping: 0.66.

That's Act 2c's subject, and it would be an embarrassing thing to trip over by
accident on camera. **Use `v11_native.model` for every demo except section 5,
which deliberately shows both.**

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

Verified working 2026-07-23.

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

Status as of 2026-07-23: the maths-only midtrain (`training/build_mathonly_corpus.py`
→ `train_mathonly.py` → `cliff_probe_mathonly.py`) is self-contained and
smoke-tested, no cell80 dependency — `train_mathonly.py` loads
`model_v11/artifacts/model_compiled.pt` and saves straight to
`model_mathonly.pt` (no vocab resize needed, this corpus never mentions a
cell). `repl.py`'s `/mathonly` command already loads the result directly. None
of the real multi-hour training runs have been fired yet.

`training/harness_pretrain/` is a from-scratch pretrain unit adapted to the
[chuk-train](https://github.com/chrishayuk/gpu-training-harness) script
contract, for running the pretrain on a GPU worker (Colab/rented) instead of
this Mac, with a live metrics dashboard.
