#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.24"]
# ///
"""Publish the Act 3 maths-only mid-training corpus to the Hub as a dataset repo.

    publish_corpus_hf.py --dry-run          # stage, measure, print the card
    publish_corpus_hf.py                    # push to chrishayuk/v11-mathonly-midtrain-corpus

The corpus is already content-addressed in the chuk-datasets catalog
(`tiny-model/mathonly-midtrain`, `ff7bf26b…`). This puts the same bytes on the
Hub under the same identity, so someone who watched the video can fetch them
without a catalog key -- and the card is written as a getting-started kit rather
than a changelog: what the rows are, how to reproduce them, and the exact
commands to mid-train on them.

Acceptance criterion: after the push, the files the Hub SERVES are downloaded to
a clean directory and that directory's chuk-datasets content_sha is recomputed.
It must equal the registered identity. Verified against the download, not the
staging dir.

WHAT THIS REFUSES TO DO, and why each is a refusal rather than a warning:

1. PUBLISH BYTES THAT ARE NOT THE REGISTERED CORPUS.
   The catalog id `ff7bf26b…` is what a checkpoint's meta.json records as
   `corpus_identity`, and it is the only join between a result and the data that
   produced it. A Hub copy that differs by one row is worse than no Hub copy:
   it makes the join silently false for everyone who takes the easy route.

2. HASH THE CORPUS WHERE IT LIVES.
   content_sha is computed over a DIRECTORY, recursively. `training/data/` also
   holds `shared_val_710.jsonl` and the whole `cells/` subtree, so hashing it in
   place gives `655400a5…` and not the corpus identity at all. Staging into a
   clean directory is therefore mandatory, not tidiness -- this is the same trap
   that made the cells corpus need its own root.

3. SILENTLY REPLACE PUBLISHED BYTES.
   If the repo already serves a different corpus this stops and says so. A
   dataset repo cited by a card, a run's meta.json and a video is a name that
   has to keep meaning one thing. --force exists because a v2 is legitimate, but
   it must be deliberate, and a v2 should almost always be a new repo id.

4. DESCRIBE THE CORPUS FROM A FILE INSTEAD OF FROM THE CORPUS.
   Every number in the card is measured from the staged bytes at publish time.
   datasets.json's composition block was written for an earlier build and had
   drifted (96,889 rows vs the registered corpus's 96,894); copying it forward
   would have published that drift as fact. The card cannot go stale relative to
   the bytes it ships with, because it is derived from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = HERE / "data"

# The corpus is exactly these two files. Anything else in the root changes the
# identity -- see refusal 2.
CORPUS_FILES = ("mathonly_corpus.jsonl", "mathonly_held_out.json")

DEFAULT_REPO = "chrishayuk/v11-mathonly-midtrain-corpus"

# The chuk-datasets identity of `tiny-model/mathonly-midtrain` v1, as registered
# (DSV-20260725-160944-00002). run_mathonly_unit.sh pins the same constant.
REGISTERED_SHA = "ff7bf26b359914344317729678884fb9fd8f1bac8e6916d67de416ab46fdf33f"
CATALOG_NAME = "tiny-model/mathonly-midtrain"
CATALOG_VERSION = "DSV-20260725-160944-00002"

BASE_REPO = "chrishayuk/v11-tinystories-115m-base"
BASE_SHA = "1841e0581574629716b646dacd4e70feaca153a8adc5ecb0b77e0e2ebdf78d9c"
TOKENIZER_REPO = "chrishayuk/v11-tokenizer"
TOKENIZER_SHA = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"
VOCAB_SIZE = 71260
CELLS_REPO = "chrishayuk/v11-cells-midtrain-corpus"
SOURCE_REPO = "https://github.com/chrishayuk/tinystories-train-video"

HUB_DATASET = "roneneldan/TinyStories"
HUB_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

BUILD_CMD = "--drill 90000 --seed 90"

# The published shared validation rows, which live in the CELLS repo because that
# arm cannot derive them (it never holds this corpus). Recorded here so the card
# can point at them and so `--check-shared-val` can prove they still come out of
# these bytes.
SHARED_VAL_FILE = "shared_val_710.jsonl"
SHARED_VAL_SHA = "e52ec44139f263653fd4b057307dad74a834c6f72d69fa77b83818e24d77f8f5"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_sha(root: Path) -> str:
    """The chuk-datasets identity, imported rather than reimplemented.

    build_mathonly_corpus.py already owns this function and prints it on every
    build; a second copy here is a second thing to keep in step, and the whole
    point of the number is that two places compute it identically.
    """
    sys.path.insert(0, str(HERE))
    from build_mathonly_corpus import content_sha as _sha
    return _sha(root)


def stage(src: Path, dest: Path) -> dict[str, str]:
    """Copy exactly the corpus files into a clean root. Refusal 2."""
    missing = [n for n in CORPUS_FILES if not (src / n).is_file()]
    if missing:
        sys.exit(
            f"\nREFUSING -- no corpus to publish. Missing from {src}:\n"
            + "".join(f"  {n}\n" for n in missing)
            + f"\nBuild it first:\n"
              f"  uv run training/build_mathonly_corpus.py {BUILD_CMD} "
              f"--expect-sha {REGISTERED_SHA}\n")
    dest.mkdir(parents=True, exist_ok=True)
    return {n: shutil.copyfile(src / n, dest / n) and sha256_file(dest / n)
            for n in CORPUS_FILES}


def _surfaces(drill: list[dict]) -> list[str]:
    """Real rows for one addition fact, so the card's surface-diversity claim is
    shown rather than illustrated.

    Matched on the addition surfaces specifically. Grouping by "the three numbers
    sum" is not enough: `1 - 1 = 0` and `1 = 0 + 1` are the same triple, so a
    naive key mixes subtraction rows into an addition example. Operands are also
    held away from 0 and 1, where a+b=c is arithmetically true but visually
    uninformative."""
    import re
    add = re.compile(r"\+| plus | more\.|That makes|in all|big ones and")
    by_fact: dict[tuple, set] = {}
    for r in drill:
        if not add.search(r["text"]):
            continue
        ns = [int(x) for x in re.findall(r"\d+", r["text"])]
        if len(ns) != 3:
            continue
        a, b, c = sorted(ns)
        if a + b == c and a > 1 and a != b and c <= 99:
            by_fact.setdefault((a, b, c), set()).add(r["text"])
    if not by_fact:
        return []
    # Most phrasings wins; ties broken towards the larger sum, which reads better
    # as an example than 2 + 3.
    best = max(by_fact.items(), key=lambda kv: (len(kv[1]), kv[0][2]))[1]
    return sorted(best, key=len)[:7]


def measure(root: Path) -> dict:
    """Every number the card asserts, computed from the staged bytes. Refusal 4.

    Drill and replay are separated by EOS, not by length. The trainer's own split
    uses `len(ids) > 40` and that heuristic misfiles 213 long narrative drill rows
    as replay -- see the card's note. EOS is exact here: build_mathonly_corpus.py
    terminates every drill row and deliberately terminates no replay row.
    """
    rows = [json.loads(l) for l in
            (root / "mathonly_corpus.jsonl").read_text().splitlines() if l.strip()]
    eos = rows[0]["ids"][-1] if rows else 3
    # Confirm EOS really is the discriminator before relying on it: the short rows
    # are unambiguously drill, and every one of them must end with the same id.
    short = [r for r in rows if len(r["ids"]) <= 40]
    if not short or len({r["ids"][-1] for r in short}) != 1:
        sys.exit("REFUSING -- cannot identify the EOS id; drill rows disagree on "
                 "their final token, so the composition table would be a guess.")
    eos = short[0]["ids"][-1]

    drill = [r for r in rows if r["ids"][-1] == eos]
    replay = [r for r in rows if r["ids"][-1] != eos]
    dt = sum(len(r["ids"]) for r in drill)
    rt = sum(len(r["ids"]) for r in replay)

    # The trainer's val slice, reproduced exactly as train_mathonly.py:289 takes it.
    ridx = [i for i, r in enumerate(rows) if len(r["ids"]) > 40]
    val = [rows[i] for i in sorted(set(ridx[-max(10, len(ridx) // 10):]))]

    held = json.loads((root / "mathonly_held_out.json").read_text())
    return {
        "surfaces": (surf := _surfaces(drill)),
        # the plain `a + b = c` phrasing of the same fact, for the prose below it
        "canonical": next((s for s in surf if s.replace(" ", "")[0].isdigit()
                           and "+" in s and s.split("=")[0].strip()[0].isdigit()
                           and "+" in s.split("=")[0]), surf[0] if surf else ""),
        "rows": len(rows), "tokens": dt + rt,
        "drill_rows": len(drill), "drill_tokens": dt,
        "drill_unique": len({r["text"] for r in drill}),
        "drill_long": len([r for r in drill if len(r["ids"]) > 40]),
        "replay_rows": len(replay), "replay_tokens": rt,
        "replay_frac": rt / (dt + rt),
        "eos": eos, "bos": rows[0]["ids"][0],
        "val_rows": len(val), "val_tokens": sum(len(r["ids"]) for r in val),
        "val_drill": len([r for r in val if r["ids"][-1] == eos]),
        "held_out": len(held["held_out"]), "tier_max": held["tier_max"],
        "max_replay_len": max((len(r["ids"]) for r in replay), default=0),
        "example": next(r["text"] for r in drill if len(r["ids"]) <= 20),
    }


def build_card(m: dict, identity: str, shas: dict[str, str], repo_id: str,
               root: Path) -> str:
    """A getting-started kit: what the rows are, how to get them, and the exact
    commands that turn them into the checkpoint the video shows."""
    total_facts = (m["tier_max"] + 1) * (m["tier_max"] + 2) // 2
    return f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - tinystories
  - arithmetic
  - mid-training
  - numeracy
  - cell-native-architectures
size_categories:
  - 10K<n<100K
---

# v11 maths-only mid-training corpus

Teach a 115M TinyStories model arithmetic **in its weights** — no tools, no
calculator, no cell calls. {m['rows']:,} rows, {m['tokens']/1e6:.2f}M tokens,
pre-tokenized.

This is one arm of a paired experiment. Its partner,
[`{CELLS_REPO}`](https://huggingface.co/datasets/{CELLS_REPO}), teaches the same
base model to **call an external tool** for the same arithmetic. The two corpora
differ by the cell content and nothing else, so the comparison is about
*absorbing* maths versus *delegating* it.

| | |
|---|---|
| Base model | [`{BASE_REPO}`](https://huggingface.co/{BASE_REPO}) (`{BASE_SHA[:12]}…`) |
| Tokenizer | [`{TOKENIZER_REPO}`](https://huggingface.co/{TOKENIZER_REPO}) (`{TOKENIZER_SHA[:12]}…`), vocab {VOCAB_SIZE:,} |
| Identity | `{identity}` |
| Catalog | `{CATALOG_NAME}` v1, `{CATALOG_VERSION}` |
| Code | [{SOURCE_REPO.split('//')[-1]}]({SOURCE_REPO}) |

**Identity** is the chuk-datasets `content_sha`: `sha256(JCS(manifest_core))` over
one shard per file, ordered by path, computed across a root holding exactly
`mathonly_corpus.jsonl` + `mathonly_held_out.json`. Every checkpoint trained on
this corpus stamps that string into its `meta.json` as `corpus_identity`, which is
the only join between a published result and the data behind it.

## Quick start

Three steps from nothing to a mid-trained model. The 12M-token run below takes
~1.3 hours on a Colab T4 and ~2.8 on an M3 (2,630 vs 1,183 tok/s, measured).
Everything declares its own dependencies inline, so there is no install step
beyond [uv](https://docs.astral.sh/uv/).

**1. Clone and fetch.** This lands the corpus *and* the base model it mid-trains,
each verified against its published identity:

```bash
git clone {SOURCE_REPO}
cd tinystories-train-video
uv run training/fetch_published.py
```

**Want only the data?** Fetch the two files by name — not with
`hf download --local-dir`:

```python
import pathlib, shutil
from huggingface_hub import hf_hub_download
d = pathlib.Path("training/data"); d.mkdir(parents=True, exist_ok=True)
for n in ("mathonly_corpus.jsonl", "mathonly_held_out.json"):
    shutil.copyfile(hf_hub_download("{repo_id}", n, repo_type="dataset"), d / n)
```

File-by-file is deliberate rather than long-winded. **The identity above is
computed over the directory, so the corpus root must hold those two files and
nothing else.** `hf download --local-dir` does not do that: even with
`--include "mathonly_*"` to keep `README.md` out, it writes a `.cache/` tree of
download metadata beside the files, and a root with that in it hashes to
`a9d67ca6…` instead. Nothing errors — training runs happily either way — but
anything that later re-derives the identity gets a different answer, which is
exactly the failure content-addressing exists to prevent.

**2. Mid-train.** {m['tokens']/1e6:.1f}M tokens is roughly one epoch; the filmed
run does 12M (~3.5 epochs) and checkpoints every 1,500 steps.

```bash
export MATHONLY_EXPECT_SHA={identity}
uv run training/train_mathonly.py \\
    --tokens 12000000 --bs 16 --lr 1e-4 --warmup 200 \\
    --seed 80 --val-every 2000 --sample-every 250 --save-every 1500
```

`MATHONLY_EXPECT_SHA` is not a check here — it is what gets stamped into each
checkpoint's `meta.json`. A mid-train checkpoint inherits its corpus rather than
producing one, so without it the weights carry no record of which corpus taught
them.

On a 16GB card, also `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The run needs ~2GB live but fragments to a 13.8GB high-water mark: training
batches of 16 replay rows × 256 tokens are 1.17GB of logits held through the
backward, and validation and sampling churn differently-shaped allocations on
top. Lowering `--bs` would fix the OOM by changing the experiment; this changes
only how the allocator lays memory out.

**3. Score it.** The two probes that decide what the run means:

```bash
uv run training/heldout_probe_mathonly.py    # facts it was never shown
uv run training/cliff_probe_mathonly.py      # in-tier vs. one digit past
```

## Row format

```json
{{"text": "{m['example']}", "ids": [{m['bos']}, ...]}}
```

Two fields, no mask — every token is supervised. (The cells arm adds a `loss`
mask, which is the difference the experiment is about.) `{m['bos']}` is BOS;
`{m['eos']}` is EOS and appears **only on drill rows**.

| species | rows | tokens | share | terminated |
|---|---|---|---|---|
| drill | {m['drill_rows']:,} | {m['drill_tokens']:,} | {1-m['replay_frac']:.1%} | all |
| replay | {m['replay_rows']:,} | {m['replay_tokens']:,} | {m['replay_frac']:.1%} | none |
| | **{m['rows']:,}** | **{m['tokens']:,}** | | |

**drill** — in-tier arithmetic over eight operations (`add`, `sub`, `mul`, `mod`,
`cmp`, `parity`, `min3`, `succ`), with `add` and `sub` drawn twice as often as the
rest. Operand ranges: add/sub 0–{m['tier_max']}, mul half times-tables and half
1-digit × 2-digit, mod 0–99 mod 2–12, and 0–999 for the comparison ops.

Each fact appears across 7–8 surfaces — 4 canonical, 3–4 narrative — with the
answer sometimes leading and the operands sometimes reversed. One fact, as it
actually appears in these rows:

```
{chr(10).join(m['surfaces'])}
```

That is load-bearing rather than decoration. With one template and fixed slots the
cheapest thing to memorise is the *string* — "the token after
`{m['canonical'].split('=')[0].strip()} =` is {m['canonical'].split('=')[-1].strip()}" —
and a held-out-fact probe would then measure template-filling rather than
arithmetic. {m['drill_rows']:,} rows carry {m['drill_unique']:,} distinct strings.

**replay** — TinyStories at pinned revision `{HUB_REVISION[:12]}`, truncated to
{m['max_replay_len']} tokens, sized to {m['replay_frac']:.1%} of the mix. Same
revision the base was pretrained on, so this is genuinely replay and not new data.
It is there to stop the model forgetting how to tell a story while it learns to
add.

## Termination is part of the corpus

Every one of the {m['drill_rows']:,} drill rows ends with EOS. No replay row does.

This is a fix, not a default. An earlier run supervised no termination at all, and
free-running generation ran on into a template-echo mode: the probe read **0.111**
free-running against **0.91/0.97** teacher-forced *on the same checkpoint*. An
order of magnitude, entirely from not knowing where to stop. Replay rows stay
unterminated because they are whole stories and the base already handles them.

## The held-out probe

`mathonly_held_out.json` lists **{m['held_out']} addition pairs withheld from every
surface in both orders** — symmetric, so commutativity cannot leak the answer.
{m['held_out']} of {total_facts:,} in-tier addition facts.

This is the decisive test and the reason the corpus exists. In-tier accuracy on
*seen* facts is consistent with a lookup table. Accuracy on facts the model was
never shown is not. `heldout_probe_mathonly.py` scores these plus the ~982 facts
that go unseen by chance.

The filmed run puts fact-unseen at **0.98**, while `cliff_probe_mathonly.py` reads
0.99/1.00/1.00/0.87 in-range collapsing to ~0 one digit past the tier — and the
base model is flat across all three bands, so the mid-train *creates* the cliff
rather than revealing one. "It memorised a table and generalised to nothing" is
false; so is "it learned addition".

## The shared validation set

Both arms score the same {m['val_rows']} held-out stories, published as
[`{SHARED_VAL_FILE}`](https://huggingface.co/datasets/{CELLS_REPO}/blob/main/{SHARED_VAL_FILE})
in the cells repo (`{SHARED_VAL_SHA[:12]}…`). It lives there because a worker
running the cells arm has no reason to hold this corpus — an early run died 31
seconds in demanding a file that was never going to exist.

Those rows are derived from **these** bytes, as the last 10% of replay rows:

```python
rows = [json.loads(l) for l in open("mathonly_corpus.jsonl")]
idx  = [i for i, r in enumerate(rows) if len(r["ids"]) > 40]
val  = [rows[i] for i in sorted(set(idx[-len(idx)//10:]))]
```

**Known wrinkle, stated because it is measurable rather than hidden:** that
`len(ids) > 40` test is a proxy for "came from TinyStories", and
{m['drill_long']} narrative drill rows are longer than 40 tokens. {m['val_drill']}
of the {m['val_rows']} validation rows are therefore arithmetic drill items rather
than stories ({m['val_drill']/m['val_rows']:.1%}). They are held out of training,
so this is not leakage — but a number reported as "story replay NLL" is measured
on {m['val_rows']-m['val_drill']} stories and {m['val_drill']} sums. Both arms
score the identical file, so the *comparison* is unaffected; the absolute value
carries that caveat. It is left as-is deliberately: changing the split would
change the file's identity and make every completed run incomparable.

## Reproducing the corpus

```bash
uv run training/build_mathonly_corpus.py {BUILD_CMD} \\
    --expect-sha {identity}
```

Bit-identical across machines — a Colab T4 and an M3 have both produced exactly
these bytes. `--expect-sha` refuses before spending any GPU time.

Determinism was not free. The replay half originally used
`ds.shuffle(seed=90, buffer_size=10000)` on a *streaming* dataset, where the seed
controls the shuffle but not what arrives to be shuffled — the reservoir depends
on shard arrival order. Two builds of the identical command gave different corpora
(1,521,190 vs 1,521,059 replay tokens), and since validation is sliced from the
replay rows, they **validated on different data** — 663 rows against 710. That, not
device arithmetic, was most of a 1.6079-vs-1.4904 val-NLL gap first blamed on
MPS-vs-CUDA. The fix is a seeded stride over the unshuffled pinned revision.

Which is the honest answer to "why publish a corpus you can rebuild": the seed was
assumed to be the identity for months, and it wasn't. Fetching bytes with a sha is.

## Files

| file | bytes | sha256 |
|---|---|---|
{chr(10).join(f'| `{n}` | {(root / n).stat().st_size:,} | `{s[:24]}…` |' for n, s in shas.items())}

## Citation

Built for a video on training a small language model from scratch and then
teaching it maths three ways. Corpus, trainer, probes and the paired cells arm are
all in [{SOURCE_REPO.split('//')[-1]}]({SOURCE_REPO}); the drill generator is a
port of CN-7's `s1_item` from the cell-native-architectures programme, minus the
cell-call machinery.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="where the built corpus lives (default training/data)")
    ap.add_argument("--expect-sha", default=REGISTERED_SHA,
                    help="refuse unless the staged corpus has this identity. "
                         "Pass an empty string only to publish a corpus that is "
                         "not the registered one, and expect to explain why.")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace bytes already published under this repo id")
    args = ap.parse_args()

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="v11-mathonly-corpus-"))
    shas = stage(args.data_dir, out_dir)

    # --- guard 1 + 2: the staged root's identity, not the source dir's ---------
    identity = content_sha(out_dir)
    if args.expect_sha and identity != args.expect_sha:
        in_place = content_sha(args.data_dir)
        sys.exit(
            f"\nREFUSING TO PUBLISH -- staged corpus is not the registered one.\n"
            f"  expected {args.expect_sha}\n  staged   {identity}\n\n"
            f"(For reference, {args.data_dir} hashed in place is {in_place} --\n"
            f"if that is what you expected, note the identity is over a root holding\n"
            f"ONLY {' + '.join(CORPUS_FILES)}.)\n\n"
            f"Rebuild it:\n"
            f"  uv run training/build_mathonly_corpus.py {BUILD_CMD} "
            f"--expect-sha {args.expect_sha}\n")

    m = measure(out_dir)
    card = build_card(m, identity, shas, args.repo_id, out_dir)
    (out_dir / "README.md").write_text(card)

    print(f"staged {args.repo_id}")
    print(f"  identity         {identity}")
    print(f"                   == registered {CATALOG_NAME} v1"
          if identity == REGISTERED_SHA else "                   (UNREGISTERED)")
    print(f"  rows             {m['rows']:,}  ({m['tokens']:,} tokens)")
    print(f"  drill            {m['drill_rows']:,} rows / {m['drill_tokens']:,} tok, "
          f"{m['drill_unique']:,} distinct strings, all EOS-terminated")
    print(f"  replay           {m['replay_rows']:,} rows / {m['replay_tokens']:,} tok "
          f"({m['replay_frac']:.2%}), none terminated")
    print(f"  held out         {m['held_out']} addition pairs, symmetric")
    print(f"  shared val       {m['val_rows']} rows, of which {m['val_drill']} are drill "
          f"(the len>40 proxy; {m['drill_long']} drill rows exceed it)")
    for n, s in shas.items():
        print(f"  {n:24s} {s}")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    api = HfApi()

    # --- guard 3: never silently replace a published corpus --------------------
    try:
        published = {
            n: sha256_file(Path(api.hf_hub_download(
                repo_id=args.repo_id, filename=n, repo_type="dataset")))
            for n in CORPUS_FILES}
        if published == shas:
            print("  precondition     repo exists, identical bytes -- re-push updates the card only")
        elif args.force:
            print("  precondition     repo holds DIFFERENT bytes -- replacing, --force given")
        else:
            diff = "".join(f"    {n}\n      published {published[n]}\n"
                           f"      staged    {shas[n]}\n"
                           for n in CORPUS_FILES if published[n] != shas[n])
            sys.exit(
                f"\nREFUSING TO PUSH -- {args.repo_id} already serves a different corpus.\n"
                f"{diff}\n"
                f"Checkpoints stamp a corpus identity and join on this name. Publish the\n"
                f"rebuild under a new repo id, or pass --force if replacing is genuinely\n"
                f"what you want.\n")
    except (RepositoryNotFoundError, EntryNotFoundError):
        print("  precondition     repo is new")

    if args.dry_run:
        print(f"\ndry run -- nothing pushed. Staged at {out_dir}")
        print(f"\n--- README.md ({len(card):,} chars) "
              f"{'-' * 40}\n{card}")
        return

    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True,
                    private=args.private)
    commit = api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset", folder_path=str(out_dir),
        commit_message=f"Maths-only mid-train corpus, {m['rows']:,} rows, "
                       f"identity {identity[:16]}")
    revision = getattr(commit, "oid", None) or "main"
    print(f"  pushed           revision {revision}")

    verify_published(args.repo_id, revision, identity, shas)

    print(f"\npublished https://huggingface.co/datasets/{args.repo_id}")
    print(f"  identity (join on this): {identity}")
    print(f"  hub revision (retrieval coordinate only): {revision}")


def verify_published(repo_id: str, revision: str, identity: str,
                     shas: dict[str, str]) -> None:
    """Verify what the Hub SERVES, not what was staged.

    Downloads the corpus files ALONE into a clean directory and recomputes the
    identity over it -- which is both the acceptance criterion and a live
    demonstration of the `--include` the card tells readers to use. A snapshot of
    the whole repo would pull README.md in and hash to something else.
    """
    from huggingface_hub import hf_hub_download

    print("\nverifying the DOWNLOADED corpus:")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for n in CORPUS_FILES:
            p = hf_hub_download(repo_id=repo_id, filename=n, repo_type="dataset",
                                revision=revision)
            shutil.copyfile(p, root / n)
            got = sha256_file(root / n)
            if got != shas[n]:
                sys.exit(f"VERIFY FAILED: hub serves {n} as {got}, staged {shas[n]}")
            print(f"  {n:24s} sha256 matches")
        got = content_sha(root)
        if got != identity:
            sys.exit(f"VERIFY FAILED: downloaded root hashes to {got}, not {identity}")
        print(f"  content_sha      {got}")
        print(f"                   recomputed over the download, matches the registered identity")


if __name__ == "__main__":
    main()
