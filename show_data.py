#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=2.18", "sentencepiece>=0.2"]
# ///
"""Act 1c — what the model is actually trained on.

  uv run show_data.py
  uv run show_data.py --rows 5
  uv run show_data.py --tokens

Streams the SAME pinned revision of TinyStories the model was trained on
(revision f54c09f), so what's on screen is what went in.

--tokens additionally shows one story turned into the integers the model
actually sees, and works out how much of the 16M-token budget one story is.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = HERE / "tokenizer" / "v11_native.model"

# the pinned revision from train_v11_replication.py — same documents, every run
HUB_SHA = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENS_PHASE1 = 16_000_000

DIM, GREEN, BOLD, RESET = "\033[2m", "\033[92m", "\033[1m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3, help="stories to show")
    ap.add_argument("--tokens", action="store_true", help="also show tokenization")
    ap.add_argument("--skip", type=int, default=0, help="skip N stories first")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"\n{BOLD}TinyStories{RESET} — the entire education of this model")
    print(f"{DIM}roneneldan/TinyStories, revision {HUB_SHA[:12]} (pinned){RESET}\n")

    ds = load_dataset("roneneldan/TinyStories", split="train",
                      streaming=True, revision=HUB_SHA)

    stories = []
    for i, row in enumerate(ds):
        if i < args.skip:
            continue
        stories.append(row["text"].strip())
        if len(stories) >= args.rows:
            break

    for i, text in enumerate(stories, 1):
        print(f"  {DIM}── story {i + args.skip} {'─' * 56}{RESET}")
        for line in text.split("\n"):
            print(f"  {line}")
        print()

    print(f"{DIM}  Synthetic. Deliberately tiny vocabulary. Written to answer:")
    print(f"  how small can a language model be and still write real English?{RESET}\n")

    if not args.tokens:
        return

    import sentencepiece as spm
    if not TOKENIZER.exists():
        sys.exit(f"missing tokenizer: {TOKENIZER}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))

    text = stories[0]
    ids = sp.encode(text)

    print(f"{BOLD}  And here is that first story as the model sees it{RESET}\n")
    head = text[:110].replace("\n", " ")
    print(f"  {DIM}text  {RESET}{head}…\n")
    print(f"  {DIM}pieces{RESET} {[sp.id_to_piece(i) for i in ids[:18]]}…\n")
    print(f"  {DIM}ids   {RESET}{ids[:18]}…\n")

    n = len(ids)
    print(f"  one story = {BOLD}{n:,} tokens{RESET}"
          f" · {len(text):,} characters"
          f" · {n/len(text):.3f} tokens/char")
    print(f"  phase 1 budget = {TOKENS_PHASE1:,} tokens"
          f" ≈ {BOLD}{TOKENS_PHASE1/n:,.0f} stories{RESET}\n")


if __name__ == "__main__":
    main()
