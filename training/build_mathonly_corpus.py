#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers>=0.20", "datasets>=2.18"]
# ///
"""Self-contained maths-only (no cells) corpus builder for Act 3.

Self-contained on purpose: unlike CN-7's mixed corpus (cell80 repo, which needs
the cell80_py Rust bindings + a ~790-cell library just to compute ground-truth
answers), this arm never mentions a cell at all -- add/sub/mul/mod are four
lines of plain Python each. Building it here means filming Act 3 needs nothing
outside this folder except a TinyStories pull from HuggingFace (pinned to the
same revision every other script in this folder uses).

Two species, matching CN-7's S1/S4 (S2/S3 -- the cell-call content -- dropped
entirely, not just disabled by a flag):
  drill   : in-tier arithmetic, 50:50 canonical ("7 + 5 = 12") vs. narrative
            (TinyStories register). Tiers match CN-7's: add/sub operands <=99,
            mul times-tables <=12x12, mod dividend<=99/divisor<=12.
  replay  : TinyStories (pinned revision f54c09f, same as show_data.py /
            the pretrain), full loss, sized to hit --replay-frac
            of the mix.

Writes training/data/mathonly_corpus.jsonl -- rows of {"text", "ids"}, always
full-loss (no masked spans exist in this corpus, unlike CN-7's S2).

Run:
  uv run build_mathonly_corpus.py --smoke                 # ~1k tokens, seconds
  uv run build_mathonly_corpus.py --drill 90000 --seed 90  # real scale
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "data" / "mathonly_corpus.jsonl"
HUB_SHA = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # pinned, matches show_data.py

# Operand ceiling for add/sub. 99 is CN-7's value, and this file's S1 is otherwise
# a verbatim copy of cn7_corpus.py's s1_item -- the generator that actually worked.
#
# It was briefly narrowed to 20 on an exposures-per-fact argument: addition over
# 0-99 is 10,000 facts at ~1.8 sightings each, against times tables at 8.2, so it
# looked badly under-taught. CN-7's own result refutes that. It reached
# P-a1 >= 0.90 in-tier canonical on FRESH instances with exactly these ranges and
# exactly that 1.8 -- which table lookup cannot produce. Something interpolation-like
# happens inside the range, so the tier does not need to be memorisable, and
# narrowing it diverged from the working recipe for no gain.
TIER_MAX = 99

# SURFACE DIVERSITY, and it is load-bearing rather than decoration.
#
# With one canonical and one narrative template per operation, and the operands
# always in the same slots, the cheapest thing for the model to memorise is the
# STRING -- "the token after '7 + 5 =' is 12". A held-out-fact probe then proves
# nothing: the held-out pair fails because that exact string was never seen, which
# says nothing about whether an algorithm exists.
#
# So every fact is taught across four canonical phrasings and three-to-four
# narratives, with the answer sometimes leading and the operands sometimes reversed,
# and commutative ops (add, mul) are taught in both orders. Then a taught fact that
# survives all those surfaces is genuinely a fact in the weights, and a held-out
# fact that fails cannot be explained away as an unseen string.

# HELD-OUT ADDITION FACTS -- the decisive test, and the reason this corpus exists
# in this shape.
#
# The tier is small enough that training covers all 441 addition pairs, so the
# in-range probe band measures RECALL, not generalisation, and cannot distinguish
# "memorised a table" from "learned to add". Withholding a slice fixes that: these
# pairs are inside the taught range, the same magnitude, and appear in none of the
# eight surfaces -- so if the model answers 7+5 and fails 9+14, the only difference
# is that it saw one and not the other. That is memorisation, demonstrated rather
# than argued.
#
# At 0-99 the pair space is 5,050 and 18k add items cover perhaps 4,400 of them, so
# some pairs go unseen naturally. The deliberate hold-out is still worth having: it is
# a DEFINED set, symmetric in both orders, guaranteed absent from all eight surfaces,
# so the probe samples from it rather than having to infer what was missed.
#
# Fixed seed for reproducibility; the pairs the script's demos use (7+5, 3+4) are
# excluded from the hold-out so those beats still work.
_DEMO_PAIRS = {(5, 7), (3, 4)}


def _pick_held_out(n=250, tier_max=TIER_MAX, seed=17):
    import random as _r
    rng = _r.Random(seed)
    pairs = [(a, b) for a in range(tier_max + 1) for b in range(a, tier_max + 1)]
    pairs = [p for p in pairs if p not in _DEMO_PAIRS]
    return set(rng.sample(pairs, n))


HELD_OUT = _pick_held_out()

NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Sue", "Tim", "Amy"]
OBJS = ["apples", "shells", "stones", "berries", "buttons", "stickers", "marbles",
        "acorns", "flowers", "coins"]


def add_sat(a, b):
    return min(a + b, 65535)


def sub_sat(a, b):
    return max(a - b, 0)


def mul_sat(a, b):
    return min(a * b, 65535)


def safe_mod(a, b):
    return a % b if b else 0


def drill_item(rng):
    """One in-tier arithmetic item, canonical or narrative surface (matches
    CN-7's s1_item tiers exactly, minus the cell-call machinery: op in
    {add,sub,mul,mod,cmp,parity,min3,succ})."""
    op = rng.choice(["add", "add", "sub", "sub", "mul", "mod", "cmp", "parity", "min3", "succ"])
    if op == "add":
        # Redraw until the pair is not held out. HELD_OUT is symmetric, so a
        # held-out fact is absent in BOTH orders -- otherwise commutativity alone
        # would leak the answer and the probe would measure nothing.
        while True:
            a, b = rng.randint(0, TIER_MAX), rng.randint(0, TIER_MAX)
            if (min(a, b), max(a, b)) not in HELD_OUT:
                break
        if rng.random() < 0.5:            # commutativity: teach both orders
            a, b = b, a
        r = add_sat(a, b)
        n1, n2, o = rng.choice(NAMES), rng.choice(NAMES), rng.choice(OBJS)
        can = rng.choice([
            f"{a} + {b} = {r}",
            f"{r} = {a} + {b}",                       # answer leads
            f"{a} plus {b} is {r}",
            f"What is {a} + {b}? It is {r}.",         # operands medial
        ])
        nar = rng.choice([
            f"{n1} had {a} {o}. {n2} gave {n1} {b} more. Now {n1} has {r} {o}.",
            f"{n1} found {b} {o} and already had {a}. That makes {r} {o}.",   # b before a
            f"There were {a} {o} on the table and {b} on the floor, so {r} in all.",
            f"{n1} counted {r} {o}: {a} big ones and {b} small ones.",        # answer first
        ])
    elif op == "sub":
        a = rng.randint(1, TIER_MAX); b = rng.randint(0, a)
        r = sub_sat(a, b)
        n1, o = rng.choice(NAMES), rng.choice(OBJS)
        can = rng.choice([
            f"{a} - {b} = {r}",
            f"{r} = {a} - {b}",
            f"{a} take away {b} is {r}",
            f"What is {a} - {b}? It is {r}.",
        ])
        nar = rng.choice([
            f"{n1} had {a} {o} and lost {b}. {n1} has {r} {o} left.",
            f"{n1} gave away {b} of {a} {o}, keeping {r}.",
            f"Of the {a} {o}, {b} were gone, so {r} were left.",
            f"{n1} has {r} {o} left after losing {b} of the {a}.",
        ])
    elif op == "mul":
        # CN-7's draw: half times tables, half 1-digit x 2-digit. Both sit inside
        # the Tier A frontier ("up to 2-digit x 1-digit; times tables").
        if rng.random() < 0.5:
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        else:
            a, b = rng.randint(0, 9), rng.randint(10, 99)
        if rng.random() < 0.5:            # commutativity
            a, b = b, a
        r = mul_sat(a, b)
        o, n1 = rng.choice(OBJS), rng.choice(NAMES)
        can = rng.choice([
            f"{a} x {b} = {r}",
            f"{r} = {a} x {b}",
            f"{a} times {b} is {r}",
            f"What is {a} x {b}? It is {r}.",
        ])
        nar = rng.choice([
            f"There were {a} bags with {b} {o} in each bag. That made {r} {o} in all.",
            f"{n1} made {r} {o} by putting {b} into each of {a} boxes.",
            f"Each of the {a} rows had {b} {o}, so there were {r}.",
        ])
    elif op == "mod":
        a, b = rng.randint(0, 99), rng.randint(2, 12)
        r = safe_mod(a, b)
        can = f"{a} mod {b} = {r}"
        o = rng.choice(OBJS)
        nar = f"{a} {o} were put in rows of {b}. There were {r} {o} left over."
    elif op == "cmp":
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        if a == b:
            b += 1
        big, small = (b, a) if a < b else (a, b)
        can = f"{small} < {big}"
        (n1, n2), o = rng.sample(NAMES, 2), rng.choice(OBJS)
        w = n1 if a > b else n2
        nar = f"{n1} found {a} {o} and {n2} found {b} {o}. {w} found more."
    elif op == "parity":
        a = rng.randint(0, 999)
        even = a % 2 == 0
        can = f"{a} is {'even' if even else 'odd'}"
        nar = f"{rng.choice(NAMES)} counted {a} {rng.choice(OBJS)}. {a} is an {'even' if even else 'odd'} number."
    elif op == "min3":
        xs = [rng.randint(0, 999) for _ in range(3)]
        r = min(xs)
        can = f"smallest of {xs[0]}, {xs[1]}, {xs[2]} is {r}"
        nar = f"Three piles had {xs[0]}, {xs[1]} and {xs[2]} {rng.choice(OBJS)}. The smallest pile had {r}."
    else:  # succ
        a = rng.randint(0, 998)
        r = add_sat(a, 1)
        can = f"after {a} comes {r}"
        nar = f"{rng.choice(NAMES)} counted {a}, then {r}."
    return can if rng.random() < 0.5 else nar


def content_sha(root: Path) -> str:
    """The chuk-datasets identity of a built corpus, computed the same way the
    catalog computes it: sha256(JCS(manifest_core)) over one shard per file,
    ordered by path relative to root, size and offset as decimal strings.

    Printed on every build so "deterministic" is a checkable claim rather than an
    assumption -- two machines can compare one line instead of diffing 29MB. It is
    also what `register.py verify` recomputes, so a mismatch here is a mismatch
    against the catalog.
    """
    import hashlib
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drill", type=int, default=90000, help="number of drill items")
    ap.add_argument("--replay-frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=90)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 500 drill items")
    ap.add_argument("--expect-sha", default="",
                    help="refuse unless the built corpus hashes to this chuk-datasets "
                         "content_sha. Use it on a worker so a rebuild that silently "
                         "differs fails loudly instead of training on the wrong bytes.")
    args = ap.parse_args()
    if args.smoke:
        args.drill = 500

    # The PUBLISHED v11 build, hash-verified. This used to load the retired
    # SentencePiece `v11_native.model` (vocab 71261) -- which the tokenizer
    # retirement deleted, so this script simply stopped working. The dangerous
    # version of that bug is the one where the old file is still lying around: the
    # corpus would be built in a 71261 id space, train_mathonly.py would train the
    # 71260-vocab model on it without complaint, and the result would be fluent
    # nonsense. Exactly the failure this project exists to stop shipping.
    from demo_common import V11Tokenizer
    sp = V11Tokenizer()
    rng = random.Random(args.seed)

    def encode(text):
        ids = sp.encode(text)
        if sp.bos_id() >= 0:
            ids = [sp.bos_id()] + ids
        return ids

    held_out_path = OUT.parent / "mathonly_held_out.json"
    held_out_path.parent.mkdir(parents=True, exist_ok=True)
    held_out_path.write_text(json.dumps(
        {"tier_max": TIER_MAX, "op": "add",
         "held_out": sorted(list(p) for p in HELD_OUT)}, indent=2))
    print(f"holding out {len(HELD_OUT)} of {(TIER_MAX+1)*(TIER_MAX+2)//2} addition "
          f"facts -> {held_out_path.name}")

    # TERMINATION SUPERVISION on drill rows -- the registered R1 remediation
    # ("add S1 EOS supervision"), never applied until now.
    #
    # R1's P-b failed at 0.111 free-running while teacher-forced knowledge read
    # 0.91/0.97 on the SAME checkpoint: an order of magnitude apart. The diagnosis
    # was mechanical -- drill rows carried no EOS, so nothing ever taught the model
    # to stop after an answer, and free-running generation ran on into a
    # template-echo mode ("10 10 = 1"-shaped, answer echoing an input). A weak
    # conditional survives teacher forcing and mode-collapses under greedy.
    #
    # Act 3b's demo is free-running greedy, so without this the on-camera number is
    # a model that demonstrably knows 12 and says something else. Not fixable in the
    # edit; only here.
    #
    # Drill rows only, matching the registered fix. Replay rows are whole
    # TinyStories and their termination is a separate question -- and leaving them
    # unterminated keeps story generation exactly as the base learned it.
    eos = sp.eos_id()
    rows = []
    for _ in range(args.drill):
        text = drill_item(rng)
        ids = encode(text)
        if eos >= 0:
            ids = ids + [eos]
        rows.append({"text": text, "ids": ids})
    drill_tokens = sum(len(r["ids"]) for r in rows)

    replay_target = int(drill_tokens * args.replay_frac / (1 - args.replay_frac))
    from datasets import load_dataset
    print(f"pulling TinyStories replay (pinned revision {HUB_SHA[:12]}) from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, revision=HUB_SHA)
    # NO .shuffle() here, and that is the whole determinism fix.
    #
    # `ds.shuffle(seed=..., buffer_size=10000)` on a STREAMING dataset shuffles
    # through a reservoir buffer, so the seed controls the shuffle but not what
    # arrives to be shuffled -- buffer contents depend on shard arrival order. Two
    # builds of the identical command therefore produced different replay sets
    # (1,521,190 vs 1,521,059 tokens; 96,889 vs 96,890 rows), and because the
    # trainer slices validation as the last 10% of replay rows, they validated on
    # DIFFERENT DATA (663 rows vs 710). That, not device arithmetic, was most of a
    # 1.6079-vs-1.4904 val-NLL gap first blamed on MPS-vs-CUDA.
    #
    # Streaming a pinned revision WITHOUT shuffling is deterministic: the shard
    # list and the row order within each shard are fixed by the revision. So take
    # that fixed order and select from it with our own seeded RNG -- deterministic
    # order x deterministic RNG = a reproducible sample that still varies by seed.
    replay_rng = random.Random(args.seed ^ 0x5EED)   # independent of the drill RNG
    keep_prob = 0.25                                  # ~4x oversample of the target
    replay_tokens = 0
    for ex in ds:
        if replay_rng.random() >= keep_prob:
            continue
        txt = ex["text"].strip()
        if not txt:
            continue
        ids = encode(txt)[:256]
        rows.append({"text": txt, "ids": ids})
        replay_tokens += len(ids)
        if replay_tokens >= replay_target:
            break

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    total = drill_tokens + replay_tokens
    print(f"drill: {args.drill} rows, {drill_tokens:,} tokens")
    print(f"replay: {replay_tokens:,} tokens ({replay_tokens/total:.1%} of {total:,} total)")
    print(f"wrote {OUT} ({len(rows)} rows)")

    sha = content_sha(OUT.parent)
    print(f"chuk-datasets content_sha: {sha}")
    if args.expect_sha and sha != args.expect_sha:
        raise SystemExit(
            "\nREFUSING -- the built corpus does not match the expected identity.\n"
            f"  expected {args.expect_sha}\n  built    {sha}\n\n"
            "Same command, different bytes. Do not train on this: a corpus that\n"
            "differs silently is how two runs end up validating on different\n"
            "data and their numbers get compared anyway.\n")


if __name__ == "__main__":
    main()
