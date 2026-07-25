# tinystories-train-video

Working folder for a video on training a tiny language model from scratch and
then teaching it maths three different ways.

**Arc:** pre-train TinyStories → ask it maths, it fails → mid-train maths (no
cells) → mid-train cell calls → why answer-only training can't work.

This is the code and assets behind it — training scripts, demo tools, and the
tokenizer/checkpoint config. All public.

## Layout

```
repl.py                ★ THE on-camera tool — the whole live shoot happens in here
demo_common.py         shared by repl.py + cold_open.py: tokenizer wrapper, vocab guard
cold_open.py           the same demos non-interactively, for rehearsal and checking
show_params.py         where the 115.1M parameters go (Act 1a) — also repl's /params
show_data.py           what TinyStories actually looks like (Act 1c) — also repl's /data
run_broker.sh          non-interactive broker smoke test (on camera it's repl's /broker)

tiny_model_v11/        vendored model code (3 files, from tiny-model/model/v11-core)

model_v11/               generated — export_repl_checkpoint.py puts a loadable
                         checkpoint here. Gitignored; not in a fresh clone.

tokenizer/
  tokenizer.json              ★ the PUBLISHED v11 build — the only tokenizer here

unit.toml                the chuk-train code-unit manifest (must live at repo root)
configs/
  real.json                   ★ Act 1e — the GPU run, pinned to the pre-tokenized stream
  real_raw.json               same run from raw text, tokenizing on the worker
  colab.json                  ★ the same 16M-token run with NO data: block — HF-streamed,
                                needs no catalog key and no control plane
  smoke.json                  20k tokens, no infra — HF streaming fallback

training/
  colab_pretrain_cell.py      ★ one Colab cell: pretrain 16M tokens, then publish to HF
  publish_pretrain_hf.py      ★ checkpoint → Hub model repo, with the tokenizer join enforced
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
uv run show_params.py
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
/config          the architecture — Act 1a
/params          where the 115.1M parameters go — Act 1a
/data [--tokens] TinyStories at the pinned revision — Act 1c
/loop            the training loop itself, five lines highlighted — Act 1d
/slow            ≈15 tok/s — readable on camera (default is ~75, too fast)
/greedy          most likely token every time; no randomness to blame
/sample /temp 0.8
/next <prompt>   top-10 next words as a bar chart, instead of generating
/len 60          how many tokens to generate
/full /compiled  switch checkpoint: after phase 1/2 vs after phase 3
/mathonly        switch checkpoint: maths mid-trained, no cells (Act 3; once trained)
/broker          let the model call Z80 cells — Act 4 (not built; needs 5c)
/fast /help /quit
```

**The whole live shoot runs in here** (SCRIPT.md § ONE REPL) rather than cutting
out to `bat` and one-shot scripts — hence `/config`, `/params`, `/data`, `/loop`.
The two exceptions are deliberate: Acts 1b and 2b use the published `v11` CLI,
because those are *tokenizer* questions and the point of 2b is that a viewer can
run the identical command. The rule is: **the `v11` CLI owns tokenizer questions,
the REPL owns this model.**

**No checkpoint is loaded until you generate.** Not just an optimisation — Acts
1a–1d happen before the model exists in the video's own story, so a REPL that had
to load trained weights to print the config would be quietly admitting the
ending. `/config`, `/params`, `/data` and `/loop` work in a clone with no
checkpoint in it.

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

Looking at the probabilities explains why. Measured on the published 16M-token
model, 2026-07-25:

| slot | number-word mass | top token |
|---|---|---|
| `Once upon a time there were ___` | **99.4%** | `two` 0.992 |
| `She counted the apples. There were ___` | 46.1% | `two` 0.449 |
| `Lily had three apples… Now Lily has ___` | 14.0% | `a` 0.530 |
| `Tom had two cats and one dog. Altogether he had ___` | 8.8% | `a` 0.588 |

**Three tiers, not two** — and the middle one is the interesting row. Pure idiom
("there were") is at 99.4%. A *counting* context with no arithmetic is at 46% —
the model half-knows a quantity belongs. Where arithmetic is actually required it
drops to 9–14% and reaches for `a` instead. So it isn't that the model has no idea
numbers exist; it has a partial one, and what it lacks specifically is
**combining** them.

`--slots` computes that closing line from what it measured rather than hardcoding
it, so it can't drift on a retrain. The retired checkpoint gave 99.1 / 2.5 / 1.2 /
3.4 — same ranking, a much sharper cliff — so quote these figures, not those.

## The tokenizer demos are the published CLI

They used to be `demo_tokenizer.py`. They are now the real tool, which is
strictly better on camera: a viewer runs the identical command.

```bash
cargo install v11-cli

v11 vocab --blocks                        # the assembled layout, one screen
v11 vocab --from 432 --count 10           # the ten digits
v11 encode --text "Once upon a time" --show-pieces
v11 pieces --text "157 divided by 16"
```

| Script beat | Command |
|---|---|
| Act 1b — what a token is | `v11 encode --text "Once upon a time" --show-pieces` |
| Act 2b — assembled, not discovered | `v11 vocab --blocks` |
| Act 2b — the digits | `v11 vocab --from 432 --count 10` |
| Act 2b — digits split, words don't | `v11 pieces --text …` |
| Act 1a — a third of the model is its vocabulary | `uv run show_params.py` |

`v11 vocab` was added for this (v-tokenizers `v11/cli`): a range walk plus
`--blocks`, which reports contiguous runs of same-kind pieces. The kinds are
derived from the pieces themselves rather than a hardcoded map, so the summary
stays honest if the vocabulary is ever rebuilt.

Act 1a stayed a script because it isn't a tokenizer question — it's model
arithmetic that happens to be about the vocabulary. It reads the architecture
from `training/harness_pretrain/config.json`, so it can't drift from the config
the Act 1e run uses.

`v11.vocab.bin` ships in the Hugging Face repo alongside `tokenizer.json`, so
the CLI works from a clean machine with no clone — `cargo install v11-cli`,
download that one file, done. crates.io carries code and not data, so without
it every CLI example would be unrunnable. Verified end to end against the
published copy.

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

The consequence was real: the **pre-existing** checkpoint was trained on the old
SentencePiece build (vocab 71261) and cannot be driven by the published tokenizer
— the ids mean different things. `repl.py` and `cold_open.py` refuse against it
rather than generating fluent nonsense:

```
  checkpoint/tokenizer mismatch -- refusing to generate.
    model_compiled.pt: vocab 71,261
    published v11 tokenizer: vocab 71,260
```

✅ **That is resolved — the replacement exists.** The 16M-token pretrain ran
2026-07-25 on the published tokenizer and is published as
`chrishayuk/v11-tinystories-115m-base`. Export it into place with
`training/export_repl_checkpoint.py` and every generating demo works.

Two things remain blocked, for unrelated reasons: `/compiled` (phase 3, the
frozen-FFN retrain, has not been run on this lineage) and `/broker` plus
`run_broker.sh` (Act 4's cell-call checkpoint needs rebuilding in cell80 — see
SCRIPT.md item 5c, still the long pole).

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

⚠️ **Blocked**, and this is now the long pole. Verified working 2026-07-23 against
the *retired* checkpoint; its cell80-side checkpoint was built on the old
vocabulary and has to be rebuilt before this runs again — see SCRIPT.md item 5c.

On camera this is no longer a script at all: it's `repl.py`'s `/broker`, so the
viewer watches the model emit the call live instead of reading a pre-baked
`PROMPT:`/`OUTPUT:` block. This script stays as the non-interactive smoke test.

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

Status as of 2026-07-25.

✅ **The Act 1e pretrain has run.** 16M tokens on the published tokenizer, locally
on MPS (~2h, the documented recipe unchanged: batch 4 × 256, lr 3e-4, seed 42),
published as `chrishayuk/v11-tinystories-115m-base`. That unblocks Acts 1f, 2a
and 3, which needed nothing but a checkpoint the published tokenizer can drive.
It also takes the chuk-datasets key off the critical path: nothing dispatches, so
nothing needs catalog auth.

Still to run:

1. **Act 3's maths-only midtrain** — `build_mathonly_corpus.py` →
   `train_mathonly.py` → `cliff_probe_mathonly.py`, self-contained and
   smoke-tested, ~2.5–3h. Now unblocked — its `--base-checkpoint` default moved to
   `model_full.pt`, since this lineage has no phase-3 checkpoint to start from.
2. **Phase 3** (frozen-FFN attention retrain, 8M tokens) — not strictly required,
   but Act 1f's `/full` vs `/compiled` beat needs it and Act 4c calls back to it.
3. **Act 4's cell-call midtrain**, in cell80 — the long pole, unscoped. `/broker`
   and `run_broker.sh` stay blocked until it lands.
4. **A screen-captured re-run of the pretrain.** The 2026-07-25 run was not
   recorded, and Act 1e needs the footage. `seed 42` plus the pinned revision make
   it reproduce exactly. See SCRIPT.md § PRE-RECORD.

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

### The Colab route — one cell, no infrastructure, publishes to the Hub

`training/colab_pretrain_cell.py` pastes into a single Colab cell on a T4 and does
the whole thing: clone, pretrain 16M tokens, publish the result to
`chrishayuk/v11-tinystories-115m-base`. ~45–75 min.

It uses `configs/colab.json`, which carries **no `data:` block**, so `train.py`
streams TinyStories straight from the pinned HF revision. That deliberately
sidesteps the blocked chuk-datasets key — this route needs no catalog key, no
control plane and no join token. What it gives up is stated in the config: the
tokenizer runs on the worker, and token *order* comes from the seed rather than a
content-addressed artifact, so the run is reproducible-given-the-seed but not
bit-reproducible the way `configs/real.json` is. Two of the four refusals also
cannot fire without staged shards — the tokenizer-sha and vocab-size guards still
do, and corpus identity rests on the HF revision pin instead.

Rehearse with `CONFIG = "configs/smoke.json"` and `PUBLISH = False` first: same
code path, 19 steps, ~2 minutes.

`training/publish_pretrain_hf.py` is the publish half, usable on its own against
any checkpoint dir `train.py` wrote. It refuses to publish a checkpoint whose
`meta.json` `tokenizer_hash` disagrees with the tokenizer being shipped beside it,
an embedding table that disagrees with the vocabulary, a step dir with no `.ready`
marker, or different weights under an already-published repo id (without
`--force`). Then it downloads what the Hub actually serves, builds `TinyModel`
from the shipped code, loads the weights and **generates** — because a shape check
passes happily through a model that emits garbage.

```bash
uv run training/publish_pretrain_hf.py --ckpt-dir out/ckpt \
    --repo-id chrishayuk/v11-tinystories-115m-base --dry-run
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
streamed, loss 10.9 → 8.0. All four refusal paths verified.

The **HF-streaming** path (`configs/colab.json`'s route) is separately verified on
MPS: 19 steps, first logged loss **11.17** — which is `ln(71260)` to two decimals,
the number Act 1d promises — down to 9.69, three milestone checkpoints written,
and `publish_pretrain_hf.py` staged a complete repo from that real output with the
tokenizer join checked. All four of its refusals were tested too.

**Neither 16M-token run has been fired yet.** The catalog-backed dispatch
(`configs/real.json`) still needs a read-scoped chuk-datasets key, since only
`/v1/datasets` is open without auth. The Colab route does not, and is the way to
get the base model built and published today.
