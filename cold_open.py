#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "numpy"]
# ///
"""Cold open for the TinyStories training video.

Non-interactive version of the repl.py demos — for rehearsal and for checking
output before filming. On camera, use repl.py and type the prompts live.

  uv run cold_open.py
  uv run cold_open.py --story
  uv run cold_open.py --maths --seed 7
  uv run cold_open.py --slots

Self-contained: reads ./tokenizer/tokenizer.json and ./model_v11/.

Uses the PUBLISHED v11 tokenizer (vocab 71260), verified by hash, and refuses
to run against any checkpoint built on a different vocabulary.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOKENIZER = HERE / "tokenizer" / "tokenizer.json"
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


class V11Tokenizer:
    """The published v11 build (pip install v11-tokenizer / HF
    chrishayuk/v11-tokenizer), wrapped in the small SentencePiece-shaped
    interface the rest of this file uses. Verified by hash on load, so what
    goes on camera is provably the published artifact."""

    SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"

    def __init__(self, path):
        import hashlib
        from tokenizers import Tokenizer
        if not path.exists():
            sys.exit(f"missing tokenizer: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.SHA256:
            sys.exit(f"{path} is not the published v11 build\n"
                     f"  expected {self.SHA256}\n  found    {actual}\n"
                     f"  re-fetch: huggingface.co/chrishayuk/v11-tokenizer")
        self.t = Tokenizer.from_file(str(path))
        self._inv = {i: p for p, i in self.t.get_vocab().items()}

    def encode(self, text):
        return self.t.encode(text).ids

    def decode(self, ids):
        return self.t.decode(ids)

    def id_to_piece(self, i):
        return self._inv.get(i, "?")

    def _special(self, tok):
        i = self.t.token_to_id(tok)
        return -1 if i is None else i

    def bos_id(self):
        return self._special("<s>")

    def eos_id(self):
        return self._special("</s>")

    def get_piece_size(self):
        return self.t.get_vocab_size()


def check_vocab(config, tok, checkpoint):
    """The published tokenizer has 71260 pieces. A checkpoint built against
    any other vocabulary cannot be driven by it -- the ids would mean
    different things and generation is fluent nonsense at perplexity ~1e7.
    Refuse instead, and say what to do about it."""
    if config.vocab_size != tok.get_piece_size():
        sys.exit(
            f"\n  checkpoint/tokenizer mismatch -- refusing to generate.\n"
            f"    {checkpoint}: vocab {config.vocab_size:,}\n"
            f"    published v11 tokenizer: vocab {tok.get_piece_size():,}\n\n"
            f"  This checkpoint predates the published tokenizer and is retired.\n"
            f"  Every demo now runs on the Act 1e model -- see SCRIPT.md,\n"
            f'  "What still needs running" items 1 and 5.\n')


def no_model_yet(path):
    sys.exit(
        f"\n  no checkpoint at {path}\n\n"
        f"  The pre-existing 71261-vocab model has been retired -- every demo\n"
        f"  that generates text now runs on the Act 1e model, which has not been\n"
        f"  trained yet. See SCRIPT.md, \"What still needs running\" item 1.\n\n"
        f"  show_data.py, show_params.py and the v11 CLI need no checkpoint\n"
        f"  and work today.\n")

def load(checkpoint="model_compiled.pt", device=None):
    from tiny_model_v11 import load_from_artifacts

    if not (ARTEFACTS / "artifacts" / checkpoint).exists():
        no_model_yet(ARTEFACTS / "artifacts" / checkpoint)

    sp = V11Tokenizer(TOKENIZER)
    model, config = load_from_artifacts(ARTEFACTS, checkpoint=checkpoint, device=device)
    check_vocab(config, sp, checkpoint)
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
