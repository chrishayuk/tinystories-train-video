#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers>=0.20", "datasets>=2.18"]
# ///
"""CN-9 cells corpus -- the DELEGATING arm, paired to the maths-only arm.

Four species, every answer computed by executing a real cell rather than by
Python arithmetic, emitted as ids with a per-token loss mask:

  S1 drill      : in-tier arithmetic, 50:50 canonical ("7 + 5 = 12") : narrative.
                  Full loss. The arm shares this species with the maths-only
                  corpus so the two differ in delegation, not in drill.
  S2 delegating : word problems whose beyond-tier step emits
                  `<call> <cell> args </call>` and then has the cell-computed
                  result INJECTED with ZERO loss on its tokens. The model learns
                  when to call and never what the answer is -- that is the whole
                  hypothesis.
  S3 emission   : cell transcripts (descriptor + `a b = r ;` pairs). An answer
                  carries loss only when the (cell, args) instance is in-tier;
                  no beyond-tier answer token anywhere carries loss.
  S4 replay     : TinyStories, full loss, ~45% of the token mix, to hold
                  storytelling in place.

Ported from cell80's `cn7_corpus.py`, which produced CN-7 R1. The species logic
is that file's and is unchanged in substance. THREE things are deliberately
different, and each one is a defect being fixed rather than a preference:

1. THE TOKENIZER, which is why this port exists at all. CN-7 built on the
   RETIRED SentencePiece `v11_native.model` (vocab 71,261, sha 4ffbfc87) with
   `<call>`/`</call>` hardcoded at 71261/71262. That is a different id mapping,
   not merely a different vocabulary size, so no CN-7 number can be compared to
   anything trained on the published build. This uses the published v11
   tokenizer (vocab 71,260, sha 10dd5110) and DERIVES every special id from the
   vocabulary size, so they cannot drift silently the way a written-down id can.

2. TERMINATION. CN-7's corpus carried no EOS anywhere, and R1 registered what
   that cost: P-b collapsed to 0.111 free-running while teacher-forced knowledge
   on the same checkpoint read 0.91/0.97. Nothing taught the model to stop, so
   greedy generation ran on into template-echo. Task rows here terminate; replay
   rows deliberately do not, exactly as the maths-only corpus settled it.

3. REPLAY DETERMINISM. CN-7 loaded TinyStories unpinned and shuffled indices.
   This takes the seeded stride over a pinned revision that build_mathonly_corpus
   arrived at the hard way -- see the comment there about two builds validating
   on different data and the ~7% val-NLL gap that got blamed on MPS-vs-CUDA.

Needs cell80: every answer is signed by running the cell that computes it, via
`cell80_py.CellHost`. That dependency lives ENTIRELY here, on the build side.
Register the output content-addressed and the training workers never need it.

    uv run build_cells_corpus.py --smoke          # ~1 min, proves the pipeline
    uv run build_cells_corpus.py --expect-sha ... # a build that must reproduce
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OUT = HERE / "data" / "cells_corpus.jsonl"
TOKEN_MAP = HERE / "data" / "cells_token_map.json"

# Pinned, and the same revision the maths-only corpus and show_data.py use --
# the two arms must replay the same stories or the comparison has a second
# variable in it.
HUB_SHA = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


# ---- the tier function (cn7_corpus.py's, unchanged) --------------------------

CELL_KIND = {
    "add_sat": "add", "add3_i16": "add",
    "sub_sat": "sub", "sub_i16": "sub",
    "mul_sat": "mul", "mul_i16": "mul", "unit_mul": "mul",
    "safe_mod": "mod", "excel_mod": "mod",
    "is_lt": "cmp", "is_gt": "cmp", "is_ge": "cmp", "is_le": "cmp", "neq": "cmp",
    "is_even": "cmp", "is_odd": "cmp",
    "min": "cmp", "max": "cmp", "min3": "cmp", "max3": "cmp",
    "min_i16": "cmp", "max_i16": "cmp", "min3_i16": "cmp", "max3_i16": "cmp",
    "mode3": "cmp", "argmin3": "cmp", "argmax3": "cmp",
    "clamp": "clamp",
}


def tier_a_instance(cell: str, args: list[int]) -> bool:
    """Whether this (cell, args) instance is inside the taught tier.

    Governs TRAINING loss, and the property it exists to hold is absolute: no
    beyond-tier answer token anywhere carries loss. That is what makes a cliff at
    the tier boundary a fact about the model rather than about the supervision.
    """
    kind = CELL_KIND.get(cell)
    if kind is None:
        return False
    if kind in ("add", "sub"):
        return all(a <= 99 for a in args)
    if kind == "mul":
        a, b = sorted(args[:2])
        return (a <= 12 and b <= 12) or (a <= 9 and b <= 99)
    if kind in ("cmp", "clamp"):
        return all(a <= 999 for a in args)
    if kind == "mod":
        return args[1] <= 12 and args[0] <= 99
    return False


# ---- encoder -----------------------------------------------------------------

_MARK = re.compile(r"(<call>|</call>|⟨[a-z0-9_]+⟩)")


class Enc:
    """Marker-aware encoder over the PUBLISHED v11 tokenizer.

    `<call>`, `</call>` and every `⟨cell⟩` are single tokens that the base
    tokenizer knows nothing about, so they are spliced in by id around ordinary
    encoded text. The ids come from `special_ids()`, never from a constant.
    """

    def __init__(self, tok, cell_ids: dict[str, int], call_id: int, close_id: int):
        self.tok, self.cell_ids = tok, cell_ids
        self.call_id, self.close_id = call_id, close_id

    def seg_ids(self, seg: str) -> list[int]:
        if seg == "<call>":
            return [self.call_id]
        if seg == "</call>":
            return [self.close_id]
        if seg.startswith("⟨") and seg.endswith("⟩"):
            return [self.cell_ids[seg[1:-1]]]
        return self.tok.encode(seg)

    def encode(self, parts: list[tuple[str, int]], terminate: bool):
        """parts = [(text, loss_flag), ...] -> (text, ids, loss).

        BOS and EOS both carry loss: the model has to learn to stop, which is
        the whole point of terminating these rows at all.
        """
        text, ids, loss = [], [], []
        for t, fl in parts:
            text.append(t)
            for seg in filter(None, _MARK.split(t)):
                si = self.seg_ids(seg)
                ids.extend(si)
                loss.extend([fl] * len(si))
        if self.tok.bos_id() >= 0:
            ids, loss = [self.tok.bos_id()] + ids, [1] + loss
        if terminate and self.tok.eos_id() >= 0:
            ids, loss = ids + [self.tok.eos_id()], loss + [1]
        return "".join(text), ids, loss


def special_ids(vocab: int) -> tuple[int, int, int]:
    """`<call>`, `</call>`, and the first cell id, derived from the vocabulary.

    Written down nowhere else. CN-7 hardcoded 71261/71262 against a 71,261-piece
    vocabulary; carrying those constants onto a 71,260-piece one would have put
    `<call>` on top of a real token and produced a corpus that trains happily
    and means something different.
    """
    return vocab, vocab + 1, vocab + 2


# ---- S1: in-tier drill -------------------------------------------------------

NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Sue", "Tim", "Amy"]
OBJS = ["apples", "shells", "stones", "berries", "buttons", "stickers", "marbles",
        "acorns", "flowers", "coins"]


def oracle_val(oracle, cell, args):
    r = oracle.run(cell, args)
    assert r.get("halt") == "returned", f"signer {cell}{args} did not return"
    return r["result"]


def s1_item(rng, oracle):
    op = rng.choice(["add", "add", "sub", "sub", "mul", "mod", "cmp", "parity", "min3", "succ"])
    if op == "add":
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        r = oracle_val(oracle, "add_sat", [a, b])
        can = f"{a} + {b} = {r}"
        n1, n2, o = rng.choice(NAMES), rng.choice(NAMES), rng.choice(OBJS)
        nar = f"{n1} had {a} {o}. {n2} gave {n1} {b} more. Now {n1} has {r} {o}."
    elif op == "sub":
        a = rng.randint(1, 99); b = rng.randint(0, a)
        r = oracle_val(oracle, "sub_sat", [a, b])
        can = f"{a} - {b} = {r}"
        n1, o = rng.choice(NAMES), rng.choice(OBJS)
        nar = f"{n1} had {a} {o} and lost {b}. {n1} has {r} {o} left."
    elif op == "mul":
        if rng.random() < 0.5:
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        else:
            a, b = rng.randint(0, 9), rng.randint(10, 99)
        r = oracle_val(oracle, "mul_sat", [a, b])
        can = f"{a} x {b} = {r}"
        o = rng.choice(OBJS)
        nar = f"There were {a} bags with {b} {o} in each bag. That made {r} {o} in all."
    elif op == "mod":
        a, b = rng.randint(0, 99), rng.randint(2, 12)
        r = oracle_val(oracle, "safe_mod", [a, b])
        can = f"{a} mod {b} = {r}"
        o = rng.choice(OBJS)
        nar = f"{a} {o} were put in rows of {b}. There were {r} {o} left over."
    elif op == "cmp":
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        if a == b:
            b += 1
        lt = oracle_val(oracle, "is_lt", [a, b])
        big, small = (b, a) if lt == 1 else (a, b)
        can = f"{small} < {big}"
        (n1, n2), o = rng.sample(NAMES, 2), rng.choice(OBJS)
        w = n1 if a > b else n2
        nar = f"{n1} found {a} {o} and {n2} found {b} {o}. {w} found more."
    elif op == "parity":
        a = rng.randint(0, 999)
        even = oracle_val(oracle, "is_even", [a])
        can = f"{a} is {'even' if even == 1 else 'odd'}"
        nar = (f"{rng.choice(NAMES)} counted {a} {rng.choice(OBJS)}. "
               f"{a} is an {'even' if even == 1 else 'odd'} number.")
    elif op == "min3":
        xs = [rng.randint(0, 999) for _ in range(3)]
        r = oracle_val(oracle, "min3", xs)
        can = f"smallest of {xs[0]}, {xs[1]}, {xs[2]} is {r}"
        nar = (f"Three piles had {xs[0]}, {xs[1]} and {xs[2]} {rng.choice(OBJS)}. "
               f"The smallest pile had {r}.")
    else:  # succ
        a = rng.randint(0, 998)
        r = oracle_val(oracle, "add_sat", [a, 1])
        can = f"after {a} comes {r}"
        nar = f"{rng.choice(NAMES)} counted {a}, then {r}."
    return [(can if rng.random() < 0.5 else nar, 1)], {"op": op}


# ---- S2: delegating word problems --------------------------------------------

# Beyond-tier steps. The `tail` never restates the result -- if it did, the model
# could recover the answer from context and the zero-loss injection would be
# teaching it the number after all.
S2_BEYOND = [
    ("mul_sat", lambda rng: [rng.randint(13, 99), rng.randint(13, 99)],
     lambda a: f"The truck brought {a[0]} crates with {a[1]} apples in each crate. "
               f"The counting machine worked it out: ",
     " apples in all. Everyone cheered."),
    ("safe_div", lambda rng: [rng.randint(100, 999), rng.randint(3, 19)],
     lambda a: f"{a[0]} sweets were shared fairly between {a[1]} children. "
               f"The sharing machine said each child gets ",
     " sweets. The children smiled."),
    ("ceil_div", lambda rng: [rng.randint(100, 999), rng.randint(6, 24)],
     lambda a: f"{a[0]} books had to go in boxes of {a[1]}. "
               f"The packing machine counted the boxes needed: ",
     " boxes. Off they went."),
    ("add_sat", lambda rng: [rng.randint(100, 999), rng.randint(100, 999)],
     lambda a: f"One field grew {a[0]} pumpkins and the other grew {a[1]}. "
               f"The farm machine added them up: ",
     " pumpkins altogether. What a harvest."),
    ("sub_sat", lambda rng: [rng.randint(500, 999), rng.randint(100, 499)],
     lambda a: f"The shop had {a[0]} balloons and sold {a[1]}. "
               f"The till machine counted what was left: ",
     " balloons stayed in the shop."),
    ("round_to_multiple", lambda rng: [rng.randint(100, 999), rng.choice([25, 50, 100])],
     lambda a: f"About {a[0]} people came to the fair. Rounded to the nearest {a[1]}, "
               f"the sign machine wrote ",
     " visitors. The mayor was proud."),
]


def s2_item(rng, oracle):
    parts = []
    if rng.random() < 0.6:            # in-tier warm-up, full loss
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        r = oracle_val(oracle, "add_sat", [a, b])
        n = rng.choice(NAMES)
        parts.append((f"{n} picked {a} berries and then {b} more, so {n} had {r} berries. ", 1))
    cell, gen, story, tail = S2_BEYOND[rng.randrange(len(S2_BEYOND))]
    args = gen(rng)
    res = oracle_val(oracle, cell, args)
    parts.append((story(args), 1))
    parts.append((f"<call> ⟨{cell}⟩ {' '.join(map(str, args))} </call> ", 1))
    parts.append((str(res), 0))       # environment-injected, ZERO loss
    parts.append((tail, 1))
    return parts, {"cell": cell, "args": args, "res": res}


# ---- S3: emission transcripts -------------------------------------------------

RANGES = [(0, 10), (0, 10), (0, 100), (0, 100), (0, 1000), (0, 1000)]


def s3_item(rng, oracle, lib, describe, name):
    arity = lib[name]["arity"]
    pairs = []
    for lo, hi in RANGES:
        for _ in range(12):
            a = [rng.randint(lo, hi) for _ in range(arity)]
            r = oracle.run(name, a)
            if r.get("halt") == "returned":
                pairs.append((a, r["result"]))
                break
        else:
            return None
    rng.shuffle(pairs)
    parts = [(f"{describe(name, lib[name]['pack'])} <call>", 1)]
    for i, (a, o) in enumerate(pairs):
        parts.append((f" {' '.join(map(str, a))} =", 1))
        parts.append((f" {o}", 1 if tier_a_instance(name, a) else 0))
        parts.append((" ;" if i < len(pairs) - 1 else " </call>", 1))
    return parts, {"cell": name, "pairs": [[a, o] for a, o in pairs]}


# ---- identity -----------------------------------------------------------------

def content_sha(root: Path) -> str:
    """The chuk-datasets identity, computed the way the catalog computes it.

    Same function as build_mathonly_corpus.py: sha256(JCS(manifest_core)) over
    one shard per file, ordered by path relative to root.
    """
    files = sorted((p for p in root.rglob("*") if p.is_file()),
                   key=lambda p: str(p.relative_to(root)))
    shards, off = [], 0
    for f in files:
        b = f.read_bytes()
        shards.append({"sha256": hashlib.sha256(b).hexdigest(),
                       "size": str(len(b)), "offset": str(off)})
        off += len(b)
    core = {"schema": "chuk-manifest-core-1", "shards": shards}
    jcs = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(jcs.encode()).hexdigest()


def load_cell80(path: Path):
    """Import cell80's oracle + library. The one heavyweight dependency, and the
    reason it is worth it: every answer in this corpus is produced by executing
    the cell that computes it, not by Python doing the sum next to it. A corpus
    that teaches delegation should be authored by the things being delegated to."""
    if not (path / "cn1_corpus.py").is_file():
        sys.exit(
            f"\ncell80 not found at {path}\n"
            f"  This corpus is cell-signed: every answer comes from running a real cell.\n"
            f"  Point --cell80 at cell80/experiments/cell-native-architectures, or set\n"
            f"  CELL80_DIR. Nothing else in this repo needs it, and no training worker\n"
            f"  does -- register the built corpus and workers fetch bytes, not cells.\n")
    sys.path.insert(0, str(path))
    try:
        from cn1_corpus import Oracle, describe
    except ImportError as e:
        sys.exit(f"\ncell80 present at {path} but not importable: {e}\n"
                 f"  `cell80_py` is a compiled extension; build it before running this.\n")
    lib = {json.loads(l)["name"]: json.loads(l)
           for l in (path / "cn1_library.jsonl").read_text().splitlines() if l.strip()}
    held = {h["name"] for h in
            json.loads((path / "cn1_axis_a_heldout.json").read_text())["held_out_cells"]}
    return Oracle, describe, lib, held


def main() -> None:
    default_cell80 = Path(os.environ.get(
        "CELL80_DIR",
        HERE.parent.parent / "cell80" / "experiments" / "cell-native-architectures"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s1", type=int, default=90000)
    ap.add_argument("--s2", type=int, default=25000)
    ap.add_argument("--s3-per-cell", type=int, default=45)
    ap.add_argument("--replay-frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=90)
    ap.add_argument("--cell80", type=Path, default=default_cell80)
    ap.add_argument("--smoke", action="store_true", help="tiny build, proves the pipeline")
    ap.add_argument("--expect-sha", default="",
                    help="refuse unless the build hashes to this identity")
    args = ap.parse_args()
    if args.smoke:
        args.s1, args.s2, args.s3_per_cell = 500, 200, 2

    Oracle, describe, lib, held = load_cell80(args.cell80)

    from demo_common import V11Tokenizer
    tok = V11Tokenizer()                     # refuses unless it is the published build
    vocab = tok.t.get_vocab_size()
    call_id, close_id, cell_first = special_ids(vocab)

    # Every cell gets an id, held-out ones included: the embedding table has to
    # have a row for a cell the model has never seen invoked, or CN-1's whole
    # address-it-by-fingerprint result has nowhere to live.
    cell_ids = {n: cell_first + i for i, n in enumerate(sorted(lib))}
    extended = cell_first + len(cell_ids)
    train_cells = sorted(n for n, r in lib.items() if r["arity"] >= 1 and n not in held)

    print(f"tokenizer   published v11, vocab {vocab:,} (sha {tok.SHA256[:16]}…)")
    print(f"specials    <call>={call_id}  </call>={close_id}  cells {cell_first}..{extended - 1}")
    print(f"vocabulary  {vocab:,} -> {extended:,}  ({len(cell_ids)} cells, "
          f"{len(held)} held out of training)")

    # Progress, because every species here executes real cells and S3 walks the
    # whole training library -- a build with no output is indistinguishable from
    # a hang, and this one is slow enough to look like one.
    import time
    t0 = time.time()

    def phase(label: str) -> None:
        print(f"  [{time.time() - t0:6.1f}s] {label}", flush=True)

    rng = random.Random(args.seed)
    enc = Enc(tok, cell_ids, call_id, close_id)
    signers = sorted(set(list(CELL_KIND) + [c for c, *_ in S2_BEYOND] + train_cells))
    oracle = Oracle(signers)
    phase(f"oracle loaded, {len(signers)} cells")

    rows = []

    def emit(species, parts, meta):
        text, ids, loss = enc.encode(parts, terminate=True)
        rows.append({"species": species, "text": text, "ids": ids, "loss": loss, "meta": meta})

    for _ in range(args.s1):
        emit("s1", *s1_item(rng, oracle))
    phase(f"S1 drill {args.s1:,} rows")

    for _ in range(args.s2):
        emit("s2", *s2_item(rng, oracle))
    phase(f"S2 delegating {args.s2:,} rows")

    for i, name in enumerate(train_cells):
        made = 0
        for _ in range(args.s3_per_cell * 3):
            if made >= args.s3_per_cell:
                break
            item = s3_item(rng, oracle, lib, describe, name)
            if item:
                emit("s3", *item)
                made += 1
        if (i + 1) % 50 == 0:
            phase(f"S3 {i + 1}/{len(train_cells)} cells")
    phase(f"S3 transcripts over {len(train_cells)} cells")

    task_tokens = sum(len(r["ids"]) for r in rows)

    # Replay, taken as a seeded stride over the PINNED revision -- see
    # build_mathonly_corpus.py for what shuffling a streaming dataset cost.
    from datasets import load_dataset
    replay_target = int(task_tokens * args.replay_frac / (1 - args.replay_frac))
    print(f"pulling TinyStories replay (pinned revision {HUB_SHA[:12]})…")
    ds = load_dataset("roneneldan/TinyStories", split="train",
                      streaming=True, revision=HUB_SHA)
    replay_rng = random.Random(args.seed ^ 0x5EED)
    replay_tokens = 0
    for ex in ds:
        if replay_rng.random() >= 0.25:
            continue
        txt = ex["text"].strip()
        if not txt:
            continue
        # Replay rows do NOT terminate: their termination is a separate question
        # and leaving them alone keeps story generation as the base learned it.
        text, ids, loss = enc.encode([(txt, 1)], terminate=False)
        rows.append({"species": "s4", "text": text, "ids": ids[:256],
                     "loss": loss[:256], "meta": {}})
        replay_tokens += len(ids[:256])
        if replay_tokens >= replay_target:
            break
    phase(f"S4 replay {replay_tokens:,} tokens (target {replay_target:,})")

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_MAP.write_text(json.dumps(
        {"tokenizer_sha256": tok.SHA256, "base_vocab": vocab, "call": call_id,
         "close": close_id, "cell_first_id": cell_first, "extended_vocab": extended,
         "cells": cell_ids}, indent=1) + "\n")
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    stats = {}
    for r in rows:
        s = stats.setdefault(r["species"], {"rows": 0, "tokens": 0, "masked": 0})
        s["rows"] += 1
        s["tokens"] += len(r["ids"])
        s["masked"] += len(r["loss"]) - sum(r["loss"])
    total = task_tokens + replay_tokens
    for name in sorted(stats):
        s = stats[name]
        print(f"  {name}  {s['rows']:>7,} rows  {s['tokens']:>10,} tokens  "
              f"{s['masked']:>8,} masked")
    print(f"total  {len(rows):,} rows, {total:,} tokens, replay {replay_tokens/total:.1%}")

    # THE PROPERTY THIS ARM STANDS ON, checked rather than asserted: the injected
    # result of a delegated call carries no loss. Every S2 row must mask exactly
    # the tokens of its own result and no others -- too few and the model is being
    # taught the answer it was supposed to delegate, too many and it is being
    # denied supervision it should have had.
    #
    # Checked by re-encoding each row's recorded result and comparing counts,
    # because "we passed 0 as the flag" is a statement about the code and this is
    # a statement about the bytes that will be trained on.
    bad = []
    for r in rows:
        if r["species"] != "s2":
            continue
        want = len(tok.encode(str(r["meta"]["res"])))
        got = len(r["loss"]) - sum(r["loss"])
        if got != want:
            bad.append((r["meta"]["cell"], r["meta"]["args"], want, got))
    if bad:
        for cell, cargs, want, got in bad[:5]:
            print(f"  MASK MISMATCH {cell}{cargs}: expected {want} masked, got {got}")
        sys.exit(f"\nREFUSING: {len(bad)} of {stats['s2']['rows']:,} delegating rows do not "
                 f"mask exactly their injected result.\nThe zero-gradient injection is the "
                 f"mechanism under test; a corpus that leaks it measures nothing.\n")
    print(f"mask audit  {stats['s2']['rows']:,} delegating rows, "
          f"every injected result masked exactly")

    sha = content_sha(OUT.parent)
    print(f"chuk-datasets content_sha: {sha}")
    if args.expect_sha and sha != args.expect_sha:
        sys.exit(f"\nREFUSING: corpus identity mismatch.\n"
                 f"  expected {args.expect_sha}\n  built    {sha}\n")


if __name__ == "__main__":
    main()
