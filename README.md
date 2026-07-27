# tinystories-train-video

Train a 115M language model from scratch on TinyStories, then teach it maths two
different ways and measure which one works.

**Arc:** pre-train TinyStories → ask it maths, it fails → mid-train maths into the
weights → mid-train it to call a tool instead → why answer-only training can't work.

This is the code behind a video, and everything it consumes or produces is
published. You can reproduce any part of it without asking anyone for a key.

**What the two arms actually did** — seed 80, same base, same 12M-token budget,
same LR schedule, same 710 held-out stories:

```
absorb   (maths corpus)  story NLL 1.5893 → 1.6976   +6.8%   fails the pre-registered ≤+5%
delegate (cells corpus)  story NLL 1.5940 → 1.5694   −1.5%   better than it started
```

The absorbing arm *did* learn arithmetic — 0.98 on addition facts it was never
shown — but at a measurable cost to the thing it already knew, and it collapses
to ~0 one digit outside the range it was taught. The delegating arm gave up
nothing.

---

## Quick start

Every script declares its own dependencies inline (PEP 723), so there is no
install step, no venv and no `--with` flags. You need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/chrishayuk/tinystories-train-video
cd tinystories-train-video
```

### 1. Look around before there is a model

```bash
uv run repl.py
```

`/config`, `/params`, `/data` and `/loop` all work in a fresh clone with no
weights anywhere — the architecture, where the 115.1M parameters go, the corpus at
its pinned revision, and the training loop itself. No checkpoint is loaded until
you actually generate something.

### 2. Get the base model and the corpus

```bash
uv run training/fetch_published.py
```

607MB of weights plus a 29.5MB corpus, each verified against its published
identity before it is put in place — so a truncated download fails here rather
than three hours into a training run.

Now `uv run repl.py` generates: type *Once upon a time* and watch it stream token
by token. `/slow` makes it readable.

### 3. Watch it fail at maths

```
/slots
```

Four number slots, measured live. It doesn't answer wrongly — it mostly doesn't
answer at all, and where it does it copies the most salient number from the
prompt. [See below](#what-slots-shows-and-why-it-matters) for why row 2 of that
table is a trap.

### 4. Mid-train it on maths

Step 2 already fetched the corpus. ~1.3h on a Colab T4, ~2.8h on an M3
(2,630 vs 1,183 tok/s, measured):

```bash
export MATHONLY_EXPECT_SHA=ff7bf26b359914344317729678884fb9fd8f1bac8e6916d67de416ab46fdf33f
uv run training/train_mathonly.py \
    --tokens 12000000 --bs 16 --lr 1e-4 --warmup 200 \
    --seed 80 --val-every 2000 --sample-every 250 --save-every 1500
```

On a 16GB card also `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —
see [the OOM note](#the-16gb-oom-is-fragmentation-not-capacity).

Rehearse first with `--smoke` — 30 steps, ~1 minute, same code path.

### 5. Probe it

```bash
uv run training/heldout_probe_mathonly.py    # facts it was never shown
uv run training/cliff_probe_mathonly.py      # in-tier vs. one digit past
```

`heldout` reads **0.98** on facts withheld from every surface in both orders.
`cliff` reads 0.99/1.00/1.00/0.87 in-range, collapsing to ~0 one digit past the
tier the corpus taught — and the base model is flat across all three bands, so the
mid-train *creates* the cliff rather than revealing one.

Then `uv run repl.py` and `/mathonly` to talk to what you just trained.

### 6. The other arm

The delegating arm teaches the same base to emit a tool call instead of an answer.
Its corpus is [`chrishayuk/v11-cells-midtrain-corpus`](https://huggingface.co/datasets/chrishayuk/v11-cells-midtrain-corpus)
and its trainer is `training/train_cells.py`. It extends the vocabulary
71,260 → 72,052, so it loads through `training/load_cells_checkpoint.py` rather
than the general exporter — see that file's header for why the general one is
right to refuse it.

---

## Published artifacts

| | what |
|---|---|
| [`chrishayuk/v11-tokenizer`](https://huggingface.co/chrishayuk/v11-tokenizer) | the published v11 build, vocab 71,260, `10dd5110…` |
| [`chrishayuk/v11-tinystories-115m-base`](https://huggingface.co/chrishayuk/v11-tinystories-115m-base) | the base everything starts from, 16M tokens, `1841e058…` |
| [`chrishayuk/v11-tinystories-115m-mathonly-ckpts`](https://huggingface.co/chrishayuk/v11-tinystories-115m-mathonly-ckpts) | seed 80's 15 mid-train checkpoints, step 1500 → 22012 |
| [`datasets/chrishayuk/v11-mathonly-midtrain-corpus`](https://huggingface.co/datasets/chrishayuk/v11-mathonly-midtrain-corpus) | the absorbing arm's corpus, `ff7bf26b…` |
| [`datasets/chrishayuk/v11-cells-midtrain-corpus`](https://huggingface.co/datasets/chrishayuk/v11-cells-midtrain-corpus) | the delegating arm's corpus, `2115d6ae…`, plus the shared validation rows |

Both corpora are also in the chuk-datasets catalog (`tiny-model/mathonly-midtrain`,
`cells/cn7-midtrain-corpus`) under the same content-addressed identities. The Hub
copies exist so none of this needs a catalog key.

**Corpus identities are computed over a directory**, so fetch corpus files by name
with `hf_hub_download` rather than `hf download --local-dir` — that CLI leaves a
`.cache/` tree in the target even with `--include`, and the root then hashes to
something else. Nothing errors; the identity is just quietly wrong.

The identity is what joins a result to its data: every mid-train checkpoint stamps
it into `meta.json` as `corpus_identity`. A checkpoint inherits its corpus rather
than producing one, so without that stamp the weights carry no record of what
taught them.

## Layout

```
repl.py                ★ the whole live demo happens in here
demo_common.py         shared by repl.py + show_data.py: tokenizer wrapper, vocab guard
show_params.py         where the 115.1M parameters go — also repl's /params
show_data.py           what TinyStories actually looks like — also repl's /data
run_broker.sh          non-interactive broker smoke test (interactively it's repl's /broker)

tiny_model_v11/        vendored model code (3 files, from tiny-model/model/v11-core)
model_v11/             generated — the checkpoint layout repl.py loads. Gitignored.
tokenizer/
  tokenizer.json              ★ the PUBLISHED v11 build — the only tokenizer here

unit.toml                the chuk-train code-unit manifest (must live at repo root)
configs/
  real.json                   the GPU run, pinned to the pre-tokenized stream
  colab.json                  the same 16M-token run with NO data: block — HF-streamed,
                              needs no catalog key and no control plane
  smoke.json                  20k tokens, no infra — HF streaming fallback

training/
  build_mathonly_corpus.py    ★ absorbing arm's corpus — self-contained, no cell80 dependency
  build_cells_corpus.py       ★ delegating arm's corpus — needs cell80_py to author answers
  train_mathonly.py           ★ absorbing arm's mid-train
  train_cells.py              ★ delegating arm's mid-train (grows the embedding first)
  train_phase3.py             frozen-FFN attention retrain (not yet run on this lineage)
  heldout_probe_mathonly.py   ★ facts withheld from every surface, in both orders
  cliff_probe_mathonly.py     ★ in-range vs. one and two digits past the tier
  publish_pretrain_hf.py      ★ checkpoint → Hub model repo, with the tokenizer join enforced
  publish_corpus_hf.py        ★ corpus → Hub dataset repo; every number in the card is
                              measured from the bytes, so it cannot drift from them
  fetch_published.py          ★ start here — base model + corpus, both identity-checked
  export_repl_checkpoint.py   train.py's checkpoint layout → what the demos load
  load_cells_checkpoint.py    the same, for the one checkpoint with a different vocabulary
  replay_run.py               play a finished run's output back at a readable speed
  colab_pretrain_cell.py      ★ one Colab cell: pretrain 16M tokens, then publish to HF
  colab_rescue_cell.py        pull checkpoints off a worker the harness lost
  run_mathonly_unit.sh        chuk-train entrypoints — `midtrain`, `cells`, `phase3`
  run_cells_unit.sh
  run_phase3_unit.sh
  harness_pretrain/           ★ the pretrain code unit dispatched to a GPU
    train.py                    the run itself; guards, then trains
    config.json                 ★ the only architecture config (vocab 71260)
    tokenizer_v11/              the published tokenizer.json, byte-identical to the Hub's
    run.sh                      entrypoint (deliberately does not touch torch)

source-docs/
  CN7-prereg.md          background/citation — CN-7's own (cell80) mixed-run pre-registration
  CN7-findings.md        background/citation — results; the CN-7 section starts at "## CN-7 R1"
```

## repl.py

Type a prompt, press enter, watch it generate **token by token**. Nothing
pre-canned — every word is produced live.

```
/config          the architecture config
/params          where the 115.1M parameters go
/data [--tokens] TinyStories at the pinned revision
/loop            the training loop itself
/next <prompt>   top-10 next-word predictions instead of generating
/slots           all four number slots at once, with the summary
/greedy          always take the most likely token (deterministic)
/sample /temp 0.8
/len 60          set max tokens to generate
/full            model_full.pt      (after phase 1/2, 16M tokens) — default
/compiled        model_compiled.pt  (after phase 3, frozen FFN; not run yet)
/mathonly        model_mathonly.pt  (the absorbing arm)
/broker          let the model call Z80 cells (the delegating arm, live)
/slow /fast /clear /help /quit
```

**No checkpoint is loaded until you generate.** Not just an optimisation — the
architecture, the corpus and the training loop all exist before the model does, so
a REPL that had to load trained weights to print its own config would be
describing something it hadn't built yet.

Ctrl-C stops a generation without leaving the REPL.

### What `/slots` shows, and why it matters

It's tempting to assume the model "puts a number in, and the number is wrong."
**That is not what happens** — it doesn't produce a number at all. It narrates
straight past the slot.

Looking at the probabilities explains why. Measured on the published 16M-token
model, 2026-07-25 — a ladder, not a cliff:

| slot | number-word mass | top token |
|---|---|---|
| `Once upon a time there were ___` | **99.4%** | `two` 0.992 |
| `Lily counted her toys. One, two, ___` | **93.7%** | `three` 0.591 |
| `She counted the apples. There were ___` | 46.1% | `two` 0.449 |
| `There were five ducks. Two swam away. Now there are ___` | 10.9% | `two` 0.109 |
| `Anna is four years old. Next year she will be ___` | **0.0%** | `very` 0.155 |

**Row 2 is the interesting one, and it is a trap.** 93.7% mass with the count
sequence ranked correctly — `three > four > five > six`. By that chart the model can
count. Generate from the same prompt and it says *"three, four, four, five, five,
five, five…"* forever: each step is a local next-token guess with no counter, and
TinyStories almost never counts past five. **A metric that reads "competent" over
behaviour that plainly isn't** — the same argument the cliff probe makes later,
arriving early and for free.

**Row 4 shows the other failure mode.** It answers `two ducks` — a number, the wrong
one, and it's "two" because "two" is in the prompt. It hands back the most salient
number in context rather than computing. So "it doesn't answer wrongly, it doesn't
answer at all" is only half true: it also copies.

These five were chosen by measuring ~20 candidates on the real checkpoint. `/slots`
computes its closing line from what it measured rather than hardcoding it, so it
can't drift on a retrain — the retired checkpoint gave 99.1 / 2.5 / 1.2 / 3.4, the
same ranking with a much sharper cliff, so quote these figures and not those.

## The tokenizer

Published 2026-07-24. This is **the** tokenizer — everything here uses it and
nothing else, and every claim about it is independently checkable:

```bash
pip install v11-tokenizer     # PyPI 0.1.0
cargo add v11-core            # crates.io 0.1.0 (also v11-builder, v11-cli)
# huggingface.co/chrishayuk/v11-tokenizer
```

sha256 `10dd5110…`, vocab **71260**, byte-safe. Vendored into the pretrain code
unit at `training/harness_pretrain/tokenizer_v11/tokenizer.json`, verified
byte-identical to the Hub copy. Every script here verifies that sha256 on startup,
so an id that appears on screen is one you can reproduce from PyPI.

The tokenizer demos are the published CLI rather than a bespoke script, so you can
run the identical command:

```bash
cargo install v11-cli

v11 vocab --blocks                        # the assembled layout, one screen
v11 vocab --from 432 --count 10           # the ten digits
v11 encode --text "Once upon a time" --show-pieces
v11 pieces --text "157 divided by 16"
```

`v11 vocab` was added for this (v-tokenizers `v11/cli`): a range walk plus
`--blocks`, which reports contiguous runs of same-kind pieces. The kinds are
derived from the pieces themselves rather than a hardcoded map, so the summary
stays honest if the vocabulary is ever rebuilt.

`show_params.py` stayed a script because it isn't a tokenizer question — it's model
arithmetic that happens to be about the vocabulary. It reads the architecture from
`training/harness_pretrain/config.json`, so it can't drift from the config the
pretrain uses.

`v11.vocab.bin` ships in the Hugging Face repo alongside `tokenizer.json`, so the
CLI works from a clean machine with no clone. crates.io carries code and not data,
so without it every CLI example would be unrunnable.

### The old tokenizer is gone

`v11_native.model` and `tokenizer_committed.json` have been deleted. One tokenizer,
one vocabulary, one model lineage.

The consequence was real: a pre-existing checkpoint was trained on the old
SentencePiece build (vocab 71261) and cannot be driven by the published tokenizer —
the ids mean different things. `repl.py` refuses against it rather than generating
fluent nonsense:

```
  checkpoint/tokenizer mismatch -- refusing to generate.
    model_compiled.pt: vocab 71,261
    published v11 tokenizer: vocab 71,260
```

That is resolved: the 16M-token pretrain ran 2026-07-25 on the published tokenizer
and is what `chrishayuk/v11-tinystories-115m-base` publishes.

## How the base model was trained

**On a Mac, locally, in ~2 hours** — `configs/colab.json`, which carries **no
`data:` block**, so `train.py` streams TinyStories from the pinned HF revision and
touches no infrastructure at all:

```bash
caffeinate -i env CHUK_CONFIG=$PWD/configs/colab.json \
  CHUK_METRICS=$PWD/run_pretrain/metrics.jsonl \
  CHUK_CKPT_DIR=$PWD/run_pretrain/ckpt \
  uv run training/harness_pretrain/train.py
```

Loss **11.07 → 1.70**, 15,625 steps, 120 min at 2,218 tok/s, seed 42. The first
logged loss is `ln(71260)` to two decimals — the number a uniform distribution over
the vocabulary predicts, measured rather than asserted. Then:

```bash
uv run training/export_repl_checkpoint.py     # -> model_v11/, what repl.py loads
uv run training/publish_pretrain_hf.py --ckpt-dir run_pretrain/ckpt \
    --repo-id chrishayuk/v11-tinystories-115m-base --dry-run
```

**Rerunning that training command replays instead of retraining.** `train.py` spots
a completed local log at the same step count and stops, pointing at `replay_run.py`.
`CHUK_FORCE_RETRAIN=1` trains anyway. Never fires on a worker (the check keys off
`CHUK_RUN_ID`, which only the control plane sets).

What the local route gives up, stated in the config rather than glossed: the
tokenizer runs at train time, and token *order* comes from the seed rather than a
content-addressed artifact — reproducible given the seed, but not bit-reproducible
the way `configs/real.json` is. Two of `train.py`'s four refusals also cannot fire
without staged shards; the tokenizer-sha and vocab-size guards still do, and corpus
identity rests on the HF revision pin.

Rehearse with `configs/smoke.json` first: 19 steps, ~2 minutes, same code path and
same data source.

### The other two routes

**Colab T4** — `training/colab_pretrain_cell.py` pastes into one cell and does the
whole thing including the Hub push, in ~45–75 min. Roughly half the wall clock of
the Mac, and it needs no catalog key either.

**Dispatch to a fleet** — `configs/real.json` via `submit_run`, against
`chuk-datasets` (`tiny-model/v11-rust-tokenized-phase1 @ 67603f8e…`) and a control
plane, with a live dashboard. The only bit-reproducible route, since the token
stream is a pinned content-addressed artifact and no tokenizer runs at train time.
**Currently blocked**: the catalog key on the control plane returns 401, and rented
GPUs aren't available (`provider_offers` reports only `mock` configured).

### It refuses rather than training on nonsense

The catalog holds **two** tokenized TinyStories, one `curl` apart, under the same
`pretrain-stream` class — and a token stream is a flat array of integers with no
in-band record of which tokenizer made it. Point a run at the wrong one and it
would train perfectly happily and hand you a loss curve.

So `train.py` won't start unless it can show the bytes belong to its tokenizer.
Four refusals, all tested:

| Check | Catches |
|---|---|
| vendored tokenizer sha ≠ published | a tampered/stale code unit |
| staged `content_sha` ≠ config's pin | pointing at the wrong catalog entry |
| any token id ≥ vocab_size | a different, larger id space (e.g. the 71261 build) |
| decoded sample isn't English | the case the pin can't catch — wrong mapping, right size |

That last one is deliberately blunt: it decodes the first 256 ids and checks the
space ratio and average word length. Measured separation across six wrong mappings
— correct text is 0.208 spaces / 3.8 chars-per-word, every wrong mapping lands at
0.035–0.085 / 10.7–27.0. Thresholds sit between the clusters (0.12 and 8.0), not
next to either. An earlier version counted "plausible prose characters" and
separated by 3%; it was replaced, because a 3% margin would have passed a wrong
mapping on a slightly different corpus.

### Publishing

`training/publish_pretrain_hf.py` works against any checkpoint dir `train.py` wrote.
Five refusals: a `meta.json` `tokenizer_hash` that disagrees with the tokenizer
shipped beside it, an embedding table that disagrees with the vocabulary, a step dir
with no `.ready`, vendored model code that differs from the copy being published,
and different weights under an already-published repo id (without `--force`).

Then it downloads what the Hub actually serves, builds `TinyModel` from the shipped
code, loads the weights and **generates** — because a shape check passes happily
through a model that emits garbage.

`training/publish_corpus_hf.py` is the same idea for a corpus. It stages into a
clean directory (the identity is over a *directory*, and `training/data/` holds
more than one corpus), refuses anything that isn't the registered content_sha,
refuses to overwrite different published bytes, and measures every number in the
dataset card from the corpus itself so the card cannot drift from what ships with
it. After the push it re-downloads and recomputes the identity.

## The two mid-trains

Both start from the same published base, run the same 12M-token budget on the same
LR schedule, and score the same 710 held-out stories.

**Absorbing** (`training/data/mathonly_corpus.jsonl`, 96,894 rows / 3.38M tokens) —
55% in-tier arithmetic drill across eight operations and seven-to-eight surfaces
each, 45% TinyStories replay. Every drill row ends with EOS and no replay row does.
That termination supervision is a fix, not a default: without it, free-running
generation ran on into a template-echo mode and the probe read **0.111** free
against **0.91/0.97** teacher-forced *on the same checkpoint*.

**Delegating** (`training/data/cells/cells_corpus.jsonl`) — the same drill plus word
problems that emit a tool call, with the injected result **masked out of the loss**.
The model is supervised on when to call and how to phrase the call, never on what
527 ÷ 6 is. Every answer in it was computed by executing a real program through a
Rust cell VM over a 790-cell library.

The corpora differ by the cell content and nothing else, which is what makes the
comparison at the top of this file mean anything.

### The 16GB OOM is fragmentation, not capacity

The run needs ~2GB live (115M weights + grads + AdamW state) but its allocator
high-water mark reached 13.84 of 16.11 GB and stayed. Training batches of 16 replay
rows at 256 tokens are 4,096 positions = 1.17GB of logits held through the backward,
and `val_nll` every 2,000 steps plus `sample()` every 250 churn differently-shaped
allocations on top.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is deliberately chosen over
lowering `--bs` or capping batch positions: both of those change batch composition,
which changes gradients, which changes the experiment. This changes only how the
allocator lays memory out.

### Known wrinkle in the shared validation set

Both arms score `shared_val_710.jsonl`, derived from the maths corpus as the last
10% of replay rows. Replay rows are identified as `len(ids) > 40`, which is a proxy
for "came from TinyStories" — and 213 of the 90,000 narrative drill rows exceed it.
So **20 of the 710 validation rows are arithmetic drill items rather than stories**
(2.8%). They are held out of training, so this is not leakage, but a figure reported
as "story replay NLL" is measured on 690 stories and 20 sums.

Both arms score the identical published file, so the paired comparison above is
unaffected. It is left as-is deliberately: changing the split would change that
file's sha and make every completed run incomparable.

---

## Production notes

Everything below is about filming this, not about running it.

### The live shoot runs in the REPL

`script/SCRIPT.md` § ONE REPL — the whole shoot happens in `repl.py` rather than
cutting out to `bat` and one-shot scripts, hence `/config`, `/params`, `/data`,
`/loop`. The two exceptions are deliberate: the tokenizer beats use the published
`v11` CLI, because those are *tokenizer* questions and the point is that a viewer
can run the identical command. The rule: **the `v11` CLI owns tokenizer questions,
the REPL owns this model.**

**Filming order for the cold open:** `/slow`, type *Once upon a time*, let it
stream. Then `/greedy` and type *Lily had three apples. Tom gave her four more. Now
Lily has*. Then `/next` on the four number slots below. `/slots` prints all four at
once with the summary line — that's the tool for re-measuring after a retrain;
`/next` one at a time is the on-camera version.

**Warm the uv cache by running each script once before you film.** The first run of
each prints one `Installed N packages` line while uv resolves; every run after is
silent. `numpy` is declared so torch doesn't complain about it on screen.

### Replay the pretrain, don't re-run it

```bash
uv run training/replay_run.py run_pretrain --speed 60 --max-gap 2
```

Nobody films two hours. Replay plays a *finished* run's output back at a speed that
reads, and it is the **more** faithful option rather than a compromise: a re-run
would produce a *second* run whose checkpoints get discarded. Replay shows the run
that actually produced the published weights, with `train.log`, `metrics.jsonl` and
`ckpt/step_*/meta.json` all on disk to check against. It is not live, and must not
be narrated as if it were.

`--max-gap` is the setting that matters: the HF stream stalls 30–60s while its
shuffle buffer refills, and at 60× that is still a freeze that reads as a crash.
Runs from 2026-07-25 on get exact timing from `train_replay.jsonl`; earlier ones are
reconstructed from the step lines' cumulative tok/s.

### The closing shot

```bash
./run_broker.sh
```

```
PROMPT: 157 sweets were shared fairly between 16 children.
        The sharing machine said each child gets
  [broker] model called safe_div(157, 16) -> cell returned 9
OUTPUT: <call> ⟨safe_div⟩ 157 16 </call> 9 sweets. The children smiled.
```

On camera this is `repl.py`'s `/broker`, so the viewer watches the model emit the
call live instead of reading a pre-baked `PROMPT:`/`OUTPUT:` block. This script
stays as the non-interactive smoke test.

It reaches into the cell80 repo, because that's where the Z80 executor lives. It
wraps two traps that cost real time to rediscover, both documented in its comments:

- **Most `cell80_py` builds in the uv cache are too old** and reject `safe_div.rs`
  with "unsupported statement expression". There's no installable pin — the script
  hardcodes a build that works, plus a search loop to find another if it's evicted.
- **The default `--max-tokens 60` loops** the call and narration three times, which
  reads as a bug on camera. Pinned to 21, which lands exactly on "The children
  smiled."

### Other on-camera traps

- transformers needs `attn_implementation="eager"` under torch 2.6.0 on this Mac,
  or it aborts on MPS.
- The pinned `cell80_py` extension lives only in the uv cache and needs the
  PYTHONPATH trick.

### Still open

`ROADMAP.md` carries the full list with diagnoses. The short version, as of
2026-07-26:

1. **No run can resume.** `cells-s80` reached step 12,000 of ~12,530 and was lost
   entirely when Colab dropped the worker. Checkpoints hold weights only — no
   optimizer or scheduler state — and no trainer has a `--resume`. Until that
   exists, treat every run as all-or-nothing and prefer shorter budgets on Colab.
2. **A worker can heartbeat while its runner is dead**, and nothing notices. Cost
   1h45m of a T4 to 70 identical assign/timeout cycles while telemetry read
   `gpu_util 0`. If a queue stops advancing, check `run_events` rather than
   `list_runs`, and restart the worker.
3. **Phase 3** (frozen-FFN, 8M tokens) has not been run on this lineage, so
   `/compiled` has nothing to load and the `/full` vs `/compiled` beat is unshootable.
4. **A screen-captured re-run of the pretrain.** The 2026-07-25 run was not
   recorded. `seed 42` plus the pinned revision make it reproduce exactly.
