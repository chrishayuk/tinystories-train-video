#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "numpy"]
# ///
"""Manifold or table? Score addition by what the corpus actually taught.

    heldout_probe_mathonly.py --ckpt model_mathonly.pt
    heldout_probe_mathonly.py --curve            # every _s<step> snapshot

The cliff probe asks "does it work outside the training RANGE". This asks the
harder question inside it: **does it work on facts it never saw?** Three bands,
derived from the corpus rather than declared:

  taught        this ordered pair appears in the corpus            -> recall
  reverse-only  the reverse appears, this order does not           -> commutativity
  fact-unseen   NEITHER order appears anywhere                     -> generalisation

The third band is the one that matters, and it is ~982 facts rather than the 250
deliberately withheld -- ~42% of the 0-99 space never appears at all, so the
designated hold-out is a subset of a much larger free sample.

WHY EXACT MATCH ALONE IS NOT ENOUGH, and what this adds:

1. PER-DIGIT ACCURACY. Exact match reads 0.00 across most of a training curve and
   tells you nothing about what is forming underneath. The units digit of a + b is
   a mod-10 function that is learnable compositionally; the tens digit needs carry.
   Held-out exact 0.00 with units-digit 0.90 is a completely different finding from
   a flat zero -- it is structure forming without consolidation, and it is the
   strongest available evidence that this is not a lookup table. Free, because
   every numeral tokenizes as `_` + individual digits (verified: no two-digit
   numeral is a single piece in v11).

2. CARRY STRATIFICATION. Split each band by whether the units column carries
   (a%10 + b%10 >= 10). A table shows no gap. Anything algorithmic pays a carry
   penalty. That single contrast is more diagnostic than the aggregate rate.

3. NLL ALONGSIDE EXACT. Exact match has no resolution at the low end of the
   curve, which is exactly where the three bands should first diverge.

READING IT, in the order that matters:

  a) `rung` first. It is the highest baseline the model's answer NLL beats. While
     the TAUGHT band reads "none" or "prior", nothing else in the table means
     anything -- a fact-unseen zero is then a not-yet-learned, not a failure to
     generalise. Taught must clear `decade` before fact-unseen is interpretable.
  b) then taught-minus-fact-unseen, which is the whole question. Keep n >= 250:
     the SE on that difference is ~0.10 at n=50 and ~0.05 at n=250, and the carry
     split halves the per-cell n again.
  c) if `free` << `exact`, read `trunc` BEFORE concluding anything about
     arithmetic. High `trunc` = the model terminated before finishing the answer,
     which is a termination-policy failure, not a wrong-digit failure. R1's P-b
     read 0.111 free against 0.91/0.97 teacher-forced for exactly this class of
     reason.
  d) `units` chance is ~0.10 once the model emits a parseable digit at all, NOT 0.
     "Units running ahead of exact" only means something above 0.10. A units of
     0.00 (as on the untrained base) means it is not emitting a number yet, so
     that floor is not yet active.

WHAT A RESULT HERE DOES NOT MEAN. CN-7.5 already established that this shape of
training yields an in-range interpolator with a cliff at the corpus boundary. If
held-out tracks taught, the honest word is that it learned the 0-99 addition
*manifold*. If it does not, the honest word is *table*. Neither is "it learned
addition".
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
ARTEFACTS = HERE.parent / "model_v11"
CORPUS = HERE / "data" / "mathonly_corpus.jsonl"


def taught_pairs(corpus: Path) -> set[tuple[int, int]]:
    """Ordered add pairs that actually appear, across every canonical surface."""
    seen: set[tuple[int, int]] = set()
    for line in corpus.read_text().splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        for rx in (r"(\d+) \+ (\d+)", r"(\d+) plus (\d+)"):
            for m in re.finditer(rx, text):
                seen.add((int(m.group(1)), int(m.group(2))))
    return seen


def bands(seen: set[tuple[int, int]], hi: int = 99):
    taught, reverse_only, fact_unseen = [], [], []
    for a in range(hi + 1):
        for b in range(hi + 1):
            if (a, b) in seen:
                taught.append((a, b))
            elif (b, a) in seen:
                reverse_only.append((a, b))
            else:
                fact_unseen.append((a, b))
    return {"taught": taught, "reverse-only": reverse_only, "fact-unseen": fact_unseen}


@torch.no_grad()
def score(model, tok, device, a: int, b: int):
    """Teacher-forced stats for `a + b = r`, decomposed by answer digit.

    Returns (nll, exact, digits) where digits is a list of per-position
    correctness ordered from the LEAST significant digit, so units is always
    digits[0] regardless of how many digits the answer has.
    """
    res = min(a + b, 65535)
    full = f"{a} + {b} = {res}"
    prefix = full[: full.rindex(str(res))]

    def enc(s):
        ids = tok.encode(s)
        return [tok.bos_id()] + ids if tok.bos_id() >= 0 else ids

    pre_ids, full_ids = enc(prefix), enc(full)
    k, n = len(pre_ids), len(full_ids) - len(pre_ids)
    if n <= 0:
        return None
    x = torch.tensor([full_ids], device=device)
    lg = model(x)[0][k - 1 : k - 1 + n]
    tgt = x[0, k : k + n]
    nll = float(F.cross_entropy(lg, tgt, reduction="mean"))
    nats = float(F.cross_entropy(lg, tgt, reduction="sum"))   # nats to specify it
    hit = (lg.argmax(-1) == tgt).tolist()

    # Keep only the positions that are actually digits, so a leading metaspace
    # token cannot be mistaken for the units place.
    digit_hits = [h for h, t in zip(hit, tgt.tolist())
                  if tok.id_to_piece(int(t)).lstrip("▁").isdigit()]

    # FREE-RUNNING exact, alongside the teacher-forced one. These are different
    # measures and the difference is not cosmetic: teacher-forcing scores each
    # answer digit GIVEN the correct preceding digits, so a units-digit error is
    # forgiven at the tens place. Free-running, it propagates. Act 3b's demo is
    # free-running greedy (`/mathonly`, type a prompt), so `free` is the number
    # that predicts what the camera sees, and `exact` is the one comparable to
    # CN-7's probe -- which is teacher-forced too. Report both; if they diverge,
    # quote `free` on screen.
    # Decode as generation actually does -- stopping at EOS -- and record WHY it
    # failed, because the two reasons have different fixes.
    #
    # Adding EOS supervision cured the run-on/template-echo failure but opened a new
    # one: the model can know both digits of 12 under teacher forcing and still
    # terminate after the first when generating. That is a TERMINATION-POLICY
    # failure (fix: EOS placement, drill-row length distribution) and reads
    # identically to an arithmetic failure (fix: more exposure) unless separated.
    # Conflating them is precisely the confusion the S1-EOS patch was spent undoing.
    eos = tok.eos_id()
    gen = list(pre_ids)
    produced = 0
    stopped_early = False
    for _ in range(n):
        nxt = int(model(torch.tensor([gen], device=device))[0, -1].argmax())
        if nxt == eos:
            stopped_early = produced < n      # EOS *after* a full answer is correct
            break
        gen.append(nxt)
        produced += 1
    free = produced == n and gen[k:] == full_ids[k:]
    return (nll, all(hit), list(reversed(digit_hits)), nats, res, free, stopped_early)


def answer_baselines(corpus: Path):
    """A LADDER of nulls, each strictly harder than the last, all from the corpus.

    An unconditional p(answer) is a weaker bar than it looks: a model that has
    learned only "answers to 0-99 addition are two-digit numbers around here" beats
    it without conditioning on the operands at all. So `beats p(answer)` flips
    earlier than the event worth dating. Three rungs instead:

      prior    -log p(r)                  better than the answer prior
      length   -log p(r | n_digits(r))    better than emitting the right LENGTH
      decade   -log p(r | r // 10)        better than approximate MAGNITUDE

    They flip in that order, and their step numbers are the Act 3 beat: the model
    learns the prior, then the magnitude, then the fact. Beating `decade` is the
    first rung that cannot be explained by anything except identifying the specific
    answer -- within a decade only the units digit is left to get right.

    This also gives the step-1,250 "magnitude-aware" hunch a baseline that could
    have falsified it. It had none, which is why it was worthless.
    """
    import collections
    import math
    by_val: collections.Counter = collections.Counter()
    for line in corpus.read_text().splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        for rx in (r"(\d+) \+ (\d+) = (\d+)", r"(\d+) plus (\d+) is (\d+)",
                   r"(\d+) = (\d+) \+ (\d+)"):
            for m in re.finditer(rx, text):
                g = [int(x) for x in m.groups()]
                by_val[g[0] if rx.startswith(r"(\d+) = ") else g[2]] += 1
    total = sum(by_val.values())
    if not total:
        return None

    by_len: dict[int, collections.Counter] = {}
    by_dec: dict[int, collections.Counter] = {}
    for r, c in by_val.items():
        by_len.setdefault(len(str(r)), collections.Counter())[r] += c
        by_dec.setdefault(r // 10, collections.Counter())[r] += c

    def nats(counter, r, support):
        # Laplace over the support, so an unproduced answer is finite not inf
        k = support + 1
        return -math.log((counter.get(r, 0) + 1) / (sum(counter.values()) + k))

    def f(r: int) -> dict[str, float]:
        return {
            "prior": nats(by_val, r, len(by_val)),
            "length": nats(by_len.get(len(str(r)), collections.Counter()), r,
                           len(by_len.get(len(str(r)), ()))),
            "decade": nats(by_dec.get(r // 10, collections.Counter()), r,
                           len(by_dec.get(r // 10, ()))),
        }
    return f


def carry(a: int, b: int) -> bool:
    return (a % 10) + (b % 10) >= 10


def probe(ckpt: str, pairs_by_band, n: int, seed: int, device: str | None,
          corpus: Path = CORPUS):
    from demo_common import V11Tokenizer, check_vocab
    from tiny_model_v11 import load_from_artifacts

    model, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint=ckpt, device=device)
    model.eval()
    dev = next(model.parameters()).device
    tok = V11Tokenizer()
    check_vocab(cfg, tok, ckpt)

    rng = random.Random(seed)
    base_of = answer_baselines(corpus)
    out = {}
    for band, pairs in pairs_by_band.items():
        if not pairs:
            continue
        sample = rng.sample(pairs, min(n, len(pairs)))
        agg = {True: [], False: []}          # keyed by carry
        for a, b in sample:
            r = score(model, tok, dev, a, b)
            if r:
                agg[carry(a, b)].append(r)
        flat = agg[True] + agg[False]
        if not flat:
            continue

        def summarise(rows):
            if not rows:
                return None
            nll = sum(r[0] for r in rows) / len(rows)
            exact = sum(r[1] for r in rows) / len(rows)
            units = sum(r[2][0] for r in rows if r[2]) / max(1, sum(1 for r in rows if r[2]))
            tens = ([r[2][1] for r in rows if len(r[2]) > 1])
            model_nats = sum(r[3] for r in rows) / len(rows)
            free = sum(r[5] for r in rows) / len(rows)
            trunc = sum(r[6] for r in rows) / len(rows)
            bl = [base_of(r[4]) for r in rows] if base_of else []
            rungs = {k: (sum(x[k] for x in bl) / len(bl) if bl else float("nan"))
                     for k in ("prior", "length", "decade")}
            return {"n": len(rows), "nll": nll, "exact": exact, "units": units,
                    "tens": (sum(tens) / len(tens)) if tens else float("nan"),
                    "nats": model_nats, "free": free, "trunc": trunc, **rungs}

        out[band] = {"all": summarise(flat),
                     "carry": summarise(agg[True]),
                     "no_carry": summarise(agg[False])}
    return out


def report(label: str, res: dict):
    print(f"\n  {label}")
    print(f"    {'band':<13} {'n':>4} {'nats':>6} | {'prior':>6} {'length':>6} {'decade':>6} "
          f"| {'rung':>8} {'exact':>6} {'free':>6} {'trunc':>6} {'units':>6} {'tens':>6} {'ex¬c':>6} {'ex c':>6}")
    for band, d in res.items():
        a, c, nc = d["all"], d["carry"], d["no_carry"]
        # highest rung the model beats -- "none" means not even the answer prior
        rung = "none"
        for name in ("prior", "length", "decade"):
            if a['nats'] < a[name]:
                rung = name
        print(f"    {band:<13} {a['n']:>4} {a['nats']:>6.2f} | {a['prior']:>6.2f} "
              f"{a['length']:>6.2f} {a['decade']:>6.2f} | {rung:>8} {a['exact']:>6.2f} "
              f"{a['free']:>6.2f} {a['trunc']:>6.2f} {a['units']:>6.2f} {a['tens']:>6.2f} "
              f"{(nc['exact'] if nc else float('nan')):>6.2f} "
              f"{(c['exact'] if c else float('nan')):>6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="model_mathonly.pt")
    ap.add_argument("--curve", action="store_true",
                    help="probe every <stem>_s<step>.pt snapshot, in step order")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--n", type=int, default=250,
                    help="pairs per band. The load-bearing quantity is "
                         "taught minus fact-unseen; at n=50 the SE on that "
                         "difference is ~0.10, so a real 0.15 gap is noise. The "
                         "carry split halves n again, so keep this >=250.")
    ap.add_argument("--seed", type=int, default=4242,
                    help="distinct from the corpus and cliff-probe seeds")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seen = taught_pairs(args.corpus)
    b = bands(seen)
    print(f"corpus {args.corpus.name}: {len(seen):,} ordered add pairs taught")
    for k, v in b.items():
        print(f"  {k:<13} {len(v):>5,} ordered pairs")
    print(f"\nsampling {args.n} per band · seed {args.seed} · carry = (a%10 + b%10) >= 10")

    stem = Path(args.ckpt).stem
    if args.curve:
        snaps = sorted((ARTEFACTS / "artifacts").glob(f"{stem}_s*.pt"),
                       key=lambda p: int(re.search(r"_s(\d+)$", p.stem).group(1)))
        if not snaps:
            sys.exit(f"no {stem}_s<step>.pt snapshots under {ARTEFACTS/'artifacts'}")
        targets = [p.name for p in snaps]
        if (ARTEFACTS / "artifacts" / args.ckpt).exists():
            targets.append(args.ckpt)
    else:
        targets = [args.ckpt]

    for name in targets:
        report(name, probe(name, b, args.n, args.seed, args.device, args.corpus))
    print()


if __name__ == "__main__":
    main()
