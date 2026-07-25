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
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        r = add_sat(a, b)
        can = f"{a} + {b} = {r}"
        n1, n2, o = rng.choice(NAMES), rng.choice(NAMES), rng.choice(OBJS)
        nar = f"{n1} had {a} {o}. {n2} gave {n1} {b} more. Now {n1} has {r} {o}."
    elif op == "sub":
        a = rng.randint(1, 99); b = rng.randint(0, a)
        r = sub_sat(a, b)
        can = f"{a} - {b} = {r}"
        n1, o = rng.choice(NAMES), rng.choice(OBJS)
        nar = f"{n1} had {a} {o} and lost {b}. {n1} has {r} {o} left."
    elif op == "mul":
        if rng.random() < 0.5:
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        else:
            a, b = rng.randint(0, 9), rng.randint(10, 99)
        r = mul_sat(a, b)
        can = f"{a} x {b} = {r}"
        o = rng.choice(OBJS)
        nar = f"There were {a} bags with {b} {o} in each bag. That made {r} {o} in all."
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drill", type=int, default=90000, help="number of drill items")
    ap.add_argument("--replay-frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=90)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 500 drill items")
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

    rows = []
    for _ in range(args.drill):
        text = drill_item(rng)
        rows.append({"text": text, "ids": encode(text)})
    drill_tokens = sum(len(r["ids"]) for r in rows)

    replay_target = int(drill_tokens * args.replay_frac / (1 - args.replay_frac))
    from datasets import load_dataset
    print(f"pulling TinyStories replay (pinned revision {HUB_SHA[:12]}) from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, revision=HUB_SHA)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)
    replay_tokens = 0
    for ex in ds:
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


if __name__ == "__main__":
    main()
