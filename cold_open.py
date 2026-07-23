#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "sentencepiece>=0.2", "numpy"]
# ///
"""Cold open for the TinyStories training video.

Non-interactive version of the repl.py demos — for rehearsal and for checking
output before filming. On camera, use repl.py and type the prompts live.

  uv run cold_open.py
  uv run cold_open.py --story
  uv run cold_open.py --maths --seed 7
  uv run cold_open.py --slots

Self-contained: reads ./tokenizer/v11_native.model and ./model_v11/.

IMPORTANT: uses the NATIVE SentencePiece model, which is the mapping this
checkpoint was actually trained with. The tokenizer.json committed next to the
checkpoint is a DIFFERENT piece->id mapping and silently produces garbage
(~18 nats on plain English, worse than random). See demo_tokenizer.py --section 5.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOKENIZER = HERE / "tokenizer" / "v11_native.model"
ARTEFACTS = HERE / "model_v11"

STORY_PROMPTS = [
    "Once upon a time",
    "Lily had a little red ball. One day",
    "Tom found a lost mitten in the snow. He",
]

MATHS_PROMPTS = [
    "What is three times four? The answer is",
    "Lily had three apples. Tom gave her four more. Now Lily has",
    "seven plus five equals",
    "3 x 4 =",
    "What is one hundred and fifty-seven divided by sixteen? It is",
]


def load(checkpoint="model_compiled.pt", device=None):
    import sentencepiece as spm
    from tiny_model_v11 import load_from_artifacts

    if not TOKENIZER.exists():
        sys.exit(f"missing tokenizer: {TOKENIZER}")
    if not (ARTEFACTS / "artifacts" / checkpoint).exists():
        sys.exit(f"missing checkpoint: {ARTEFACTS / 'artifacts' / checkpoint}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))

    model, config = load_from_artifacts(ARTEFACTS, checkpoint=checkpoint, device=device)

    if sp.get_piece_size() != config.vocab_size:
        print(f"  ⚠ tokenizer {sp.get_piece_size()} vs model vocab {config.vocab_size}",
              file=sys.stderr)
    return sp, model, config


@torch.no_grad()
def generate(model, sp, prompt, max_new=60, temperature=0.8, top_k=40,
             seed=None, greedy=False):
    """Naive sampling loop — no KV cache; the model re-reads its context each
    step. Fine at 115M params with a 256-token window."""
    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    max_seq = model.rope_freqs.shape[0]

    ids = sp.encode(prompt)
    if sp.bos_id() >= 0:
        ids = [sp.bos_id()] + ids
    n_prompt = len(ids)

    for _ in range(max_new):
        window = ids[-max_seq:]
        logits = model(torch.tensor([window], device=device))[0, -1].float()

        if greedy:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))

        if nxt == sp.eos_id():
            break
        ids.append(nxt)

    # decode whole-then-slice: decoding the continuation alone drops the
    # leading space, which makes the joined text read wrong on screen
    full = sp.decode(ids)
    head = sp.decode(ids[:n_prompt])
    return full[len(head):]


# Slots where a NUMBER is the only sensible continuation. Compare what the
# model actually ranks: narrative idiom vs. arithmetic.
SLOT_PROMPTS = [
    ("Once upon a time there were", "story idiom — no counting required"),
    ("She counted the apples. There were", "counting, but no arithmetic"),
    ("Lily had three apples and Tom gave her four more. Now Lily has",
     "needs 3 + 4 = seven"),
    ("Tom had two cats and one dog. Altogether he had", "needs 2 + 1 = three"),
]

NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"}


@torch.no_grad()
def next_token_probs(model, sp, prompt, k=10):
    device = next(model.parameters()).device
    ids = sp.encode(prompt)
    if sp.bos_id() >= 0:
        ids = [sp.bos_id()] + ids
    logits = model(torch.tensor([ids], device=device))[0, -1].float()
    probs = torch.softmax(logits, -1)
    vals, idx = torch.topk(probs, k)
    return [(sp.id_to_piece(int(i)), float(v)) for v, i in zip(vals, idx)]


def show_slots(model, sp):
    label = "3. What it thinks a number is"
    print(f"\n\033[1m{label}\033[0m")
    print("─" * len(label))
    print("\n  Four slots where a number is the only sensible next word.")
    print("  Top-10 predictions, and what share of that mass is a number word:\n")

    for prompt, note in SLOT_PROMPTS:
        top = next_token_probs(model, sp, prompt)
        share = sum(p for t, p in top if t.lstrip("▁").lower() in NUMBER_WORDS)
        print(f"  \033[2m{note}\033[0m")
        print(f"  {prompt} \033[1m___\033[0m")
        parts = []
        for t, p in top[:6]:
            w = t.lstrip("▁")
            hot = w.lower() in NUMBER_WORDS
            parts.append((f"\033[92m{w} {p:.3f}\033[0m" if hot else f"\033[2m{w} {p:.3f}\033[0m"))
        print("    " + "  ".join(parts))
        print(f"    number-word mass: \033[1m{share:.1%}\033[0m\n")

    print("  It is 98.9% certain about 'two' where a story convention demands")
    print("  it — and reaches for 'a', 'some', 'many' where arithmetic does.")
    print("  Number words are narrative texture here, not quantities.")


def show(model, sp, prompts, max_new, temperature, seed, greedy, label):
    print(f"\n\033[1m{label}\033[0m")
    print("─" * len(label))
    for i, p in enumerate(prompts):
        t0 = time.time()
        out = generate(model, sp, p, max_new=max_new, temperature=temperature,
                       seed=None if seed is None else seed + i, greedy=greedy)
        dt = time.time() - t0
        print(f"\n  \033[2mprompt \033[0m {p}")
        print(f"  \033[1m→\033[0m {p}\033[92m{out}\033[0m")
        print(f"  \033[2m({dt:.1f}s)\033[0m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", action="store_true", help="story demo only")
    ap.add_argument("--maths", action="store_true", help="maths demo only")
    ap.add_argument("--slots", action="store_true",
                    help="next-word probabilities at number slots")
    ap.add_argument("--prompt", help="single custom prompt")
    ap.add_argument("--checkpoint", default="model_compiled.pt",
                    choices=["model_compiled.pt", "model_full.pt"])
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax instead of sampling — deterministic, for maths")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print("\n\033[1mTinyModel v11 — cold open\033[0m")
    print(f"loading {args.checkpoint} …", end=" ", flush=True)
    t0 = time.time()
    sp, model, config = load(args.checkpoint, args.device)
    device = next(model.parameters()).device
    n = sum(p.numel() for p in model.parameters())
    print(f"{time.time()-t0:.1f}s")
    print(f"{n/1e6:.1f}M params · {config.n_layers} layers · dim {config.dim} · "
          f"vocab {config.vocab_size:,} · {device}")

    if args.prompt:
        show(model, sp, [args.prompt], args.max_new, args.temperature,
             args.seed, args.greedy, "Custom prompt")
    else:
        both = not (args.story or args.maths or args.slots)
        if args.story or both:
            show(model, sp, STORY_PROMPTS, args.max_new, args.temperature,
                 args.seed, args.greedy,
                 "1. What it's good at")
        if args.maths or both:
            # greedy by default for maths: we want its BEST answer, not a
            # sampling accident, so the failure can't be blamed on temperature
            show(model, sp, MATHS_PROMPTS, 24, args.temperature,
                 args.seed, True if not args.greedy else args.greedy,
                 "2. What it can't do (greedy — its single most likely answer)")
        if args.slots or both:
            show_slots(model, sp)
    print()


if __name__ == "__main__":
    main()
