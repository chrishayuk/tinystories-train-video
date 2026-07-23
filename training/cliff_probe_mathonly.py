#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "sentencepiece>=0.2", "numpy"]
# ///
"""Self-contained Act 3 cliff probe -- in-range vs. one/two-digit-past accuracy.

Teacher-forced answer NLL + argmax-exact, three operand-magnitude bands per op,
on the exact drill-item canonical templates build_mathonly_corpus.py trains on:
  B0 in-range   : the training generator's own ranges
  B1 one-past   : one more decimal digit on the operand(s)
  B2 well-past  : two more
This is the "watch it fall off a cliff one digit outside its training range"
measurement -- Act 3c's centrepiece.

Run:
  uv run cliff_probe_mathonly.py --ckpt model_mathonly.pt --n 150
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TOKENIZER = HERE.parent / "tokenizer" / "v11_native.model"
ARTEFACTS = HERE.parent / "model_v11"


def add_sat(a, b):
    return min(a + b, 65535)


def sub_sat(a, b):
    return max(a - b, 0)


def mul_sat(a, b):
    return min(a * b, 65535)


def safe_mod(a, b):
    return a % b if b else 0


# op -> ([B0/B1/B2 arg generators], template, ground-truth fn)
BANDS = {
    "add": ([lambda r: (r.randint(0, 99), r.randint(0, 99)),
             lambda r: (r.randint(100, 999), r.randint(100, 999)),
             lambda r: (r.randint(1000, 9999), r.randint(1000, 9999))],
            lambda a, b, res: f"{a} + {b} = {res}", add_sat),
    "sub": ([lambda r: (lambda a: (a, r.randint(0, a)))(r.randint(1, 99)),
             lambda r: (lambda a: (a, r.randint(0, a)))(r.randint(100, 999)),
             lambda r: (lambda a: (a, r.randint(0, a)))(r.randint(1000, 9999))],
            lambda a, b, res: f"{a} - {b} = {res}", sub_sat),
    "mul": ([lambda r: (r.randint(0, 12), r.randint(0, 12)),
             lambda r: (r.randint(13, 99), r.randint(13, 99)),
             lambda r: (r.randint(100, 999), r.randint(100, 999))],
            lambda a, b, res: f"{a} x {b} = {res}", mul_sat),
    "mod": ([lambda r: (r.randint(0, 99), r.randint(2, 12)),
             lambda r: (r.randint(100, 999), r.randint(2, 12)),
             lambda r: (r.randint(1000, 9999), r.randint(2, 12))],
            lambda a, b, res: f"{a} mod {b} = {res}", safe_mod),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model_mathonly.pt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()

    from tiny_model_v11 import load_from_artifacts
    model, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint=args.ckpt, device=args.device)
    device = next(model.parameters()).device
    model.eval()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))

    def encode(text):
        ids = sp.encode(text)
        if sp.bos_id() >= 0:
            ids = [sp.bos_id()] + ids
        return ids

    rng = random.Random(985)  # distinct from build_mathonly_corpus.py's default seed 90

    @torch.no_grad()
    def answer_stats(template, a, b, res):
        full_text = template(a, b, res)
        prefix_text = full_text[:full_text.rindex(str(res))]
        prefix_ids, full_ids = encode(prefix_text), encode(full_text)
        k = len(prefix_ids)
        n = len(full_ids) - k
        if n <= 0:
            return None
        x = torch.tensor([full_ids], device=device)
        lg = model(x)[0]
        tgt = x[0, k:k + n]
        nll = float(F.cross_entropy(lg[k - 1:k - 1 + n], tgt, reduction="mean"))
        exact = bool((lg[k - 1:k - 1 + n].argmax(-1) == tgt).all())
        return nll, exact

    print(f"probing {args.ckpt} ({device})")
    for op, (gens, template, fn) in BANDS.items():
        cells = []
        for b_idx, gen in enumerate(gens):
            nlls, exacts = [], []
            tries = 0
            while len(nlls) < args.n and tries < args.n * 8:
                tries += 1
                a, b = gen(rng)
                res = fn(a, b)
                if res == 65535:  # saturation excluded -- trivially learnable, fakes robustness
                    continue
                stats = answer_stats(template, a, b, res)
                if stats is None:
                    continue
                nll, exact = stats
                nlls.append(nll); exacts.append(exact)
            cells.append((len(nlls), sum(nlls) / max(1, len(nlls)), sum(exacts) / max(1, len(exacts))))
        print(f"  {op:<6} " + "  ".join(
            f"B{i}: nll {nll:.2f} exact {exact:.2f} (n={n})" for i, (n, nll, exact) in enumerate(cells)))

    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
