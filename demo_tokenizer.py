#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers>=0.20"]
# ///
"""On-camera tokenizer demos for the TinyStories training video.

Uses the PUBLISHED v11 tokenizer -- `pip install v11-tokenizer`, crates.io
`v11-core`, HF `chrishayuk/v11-tokenizer`. The vendored ./tokenizer/tokenizer.json
is byte-identical to the Hub copy (sha256 10dd5110...), and this script checks
that on startup, so every ID it prints on camera is one a viewer can reproduce.

Self-contained: reads only from ./tokenizer/, no repo paths, no network.

  uv run demo_tokenizer.py

Sections map to the script:
  1  Act 1b  -- what a token is
  2  Act 2b  -- v11 is a knowledge-first tokenizer
  3  Act 2b  -- digits split, number words don't  (why maths is unnatural here)
  4  Act 1a  -- a third of the model is its vocabulary

Run with --section N to show just one on camera.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = HERE / "tokenizer" / "tokenizer.json"       # the published v11 build
PUBLISHED_SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"

# v11 architecture (training/harness_pretrain/config.json)
DIM, N_LAYERS, FFN_DIM, N_HEADS, N_KV_HEADS, VOCAB = 512, 20, 2048, 8, 4, 71260


def rule(title):
    print(f"\n\033[1m{title}\033[0m")
    print("─" * len(title))


def load_tokenizer():
    """The published v11 build, verified by hash -- so nothing on camera
    depends on which copy of the file happens to be lying around."""
    import hashlib
    from tokenizers import Tokenizer
    if not TOKENIZER.exists():
        sys.exit(f"missing tokenizer: {TOKENIZER}")
    actual = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
    if actual != PUBLISHED_SHA256:
        sys.exit(f"{TOKENIZER} is not the published v11 build\n"
                 f"  expected {PUBLISHED_SHA256}\n  found    {actual}\n"
                 f"  re-fetch: huggingface.co/chrishayuk/v11-tokenizer")
    return Tokenizer.from_file(str(TOKENIZER))


class Tok:
    """Thin adapter so the sections read the same as before."""
    def __init__(self, t):
        self.t = t
        self._inv = {i: p for p, i in t.get_vocab().items()}

    def encode(self, text):
        return self.t.encode(text).ids

    def id_to_piece(self, i):
        return self._inv.get(i, "?")

    def get_piece_size(self):
        return self.t.get_vocab_size()


def pieces(sp, text):
    return [sp.id_to_piece(i) for i in sp.encode(text)]


# ── 1 ── Act 1b: what a token is ─────────────────────────────────────────────
def section_1(sp):
    rule("1. What a token is")
    text = "Once upon a time"
    print(f'  t.encode("{text}").tokens')
    print(f"  → {pieces(sp, text)}")
    print()
    print("  Four tokens. '▁' just marks where a space was.")
    print("  The model never sees words — it sees a list of integers:")
    print(f"  → {sp.encode(text)}")


# ── 2 ── Act 2b: knowledge-first vocabulary ──────────────────────────────────
# Real ID offsets in the published v11 vocabulary, read off the artifact --
# not guessed. The first ~700 IDs are an entirely hand-placed character and
# symbol prelude; the first English word appears around 1000.
BLOCKS = [
    (0, "special tokens"),
    (4, "byte fallback — all 256 byte values, so nothing is unrepresentable"),
    (260, "every letter, bare and space-prefixed"),
    (312, "punctuation"),
    (343, "capitals"),
    (395, "multi-character operators"),
    (432, "the ten digits — one piece each, never bundled into number-chunks"),
    (442, "Greek, lower then upper"),
    (573, "fractions"),
    (592, "logic, maths and set operators"),
    (698, "the number sets"),
    (1000, "curated morphemes"),
    (8000, "tree-sitter AST node types — the grammars of 77 languages"),
    (20000, "ordinary English"),
    (40000, "WordNet's long tail"),
]


def section_2(sp):
    rule("2. v11 is assembled, not discovered")
    n = sp.get_piece_size()
    print(f"  vocabulary size: {n:,} pieces")
    print()
    print("  Most tokenizers are DISCOVERED: run BPE over a corpus, keep whatever")
    print("  chunks compress best. Nobody chooses the vocabulary.")
    print()
    print("  v11 is ASSEMBLED — from WordNet, Wikidata, tree-sitter grammars for")
    print("  77 programming languages, curated morphemes, Greek letters, maths")
    print("  symbols and acronyms. You can SEE that, because the sources sit in")
    print("  contiguous blocks. Walk the IDs in order:")
    print()
    for lo, label in BLOCKS:
        ps = [sp.id_to_piece(i) for i in range(lo, min(lo + 8, n))]
        print(f"    {lo:>6}  {' '.join(ps)}")
        print(f"            \033[2m{label}\033[0m")
    print()
    print("  That is not a statistical artefact. That is a hand-built index.")
    print()
    print("  Design principle, from the v11 README, written long before any")
    print("  experiment in this video existed:")
    print("      'Every token is a potential compilation target.'")
    print()
    print("  Note the first ~700 IDs: bytes, letters, punctuation, operators,")
    print("  digits, Greek, fractions, maths, set theory. Not one is a WORD.")
    print("  The digits sit at 432-441 — one piece per digit, before any English.")


# ── 3 ── Act 2b: digits split, number words don't ────────────────────────────
def section_3(sp):
    rule("3. Why maths is unnatural here")
    print("  Number WORDS are single tokens:")
    for w in ["zero", "three", "seven", "twelve", "sixteen", "twenty", "hundred"]:
        p = pieces(sp, " " + w)
        print(f"    {w:<10} {len(p)} token   {p}")

    print("\n  DIGITS are split one per token:")
    for s in ["12", "157", "1234"]:
        print(f"    {s:<10} {len(pieces(sp, s))} tokens  {pieces(sp, s)}")

    print("\n  So the same quantity is words vs. bare digit characters:")
    print(f"    'one hundred and fifty-seven' → {pieces(sp, 'one hundred and fifty-seven')}")
    print(f"    '157'                         → {pieces(sp, '157')}")

    print("\n  Side by side:")
    for s in ["three times four is twelve", "3 x 4 = 12", "157 divided by 16"]:
        p = pieces(sp, s)
        print(f"    {s:<30} {len(p):>2} tokens  {p}")

    print("\n  Reading '157 divided by 16', the model does not see two quantities.")
    print("  It sees a string of digit characters. To answer it would have to")
    print("  learn place value and carrying as SPELLING PATTERNS, with no notion")
    print("  that these symbols denote magnitude at all.")
    print()
    print("  This is the sane choice — the alternative is a vocabulary full of")
    print("  arbitrary number-chunks. But it makes arithmetic an odd thing to learn.")


# ── 4 ── Act 1a: a third of the model is its vocabulary ──────────────────────
def section_4():
    rule("4. A third of the model is its vocabulary")
    head_dim = DIM // N_HEADS
    emb = VOCAB * DIM
    attn = DIM * DIM * 2 + DIM * N_KV_HEADS * head_dim * 2
    ffn = 3 * DIM * FFN_DIM
    layers = N_LAYERS * (attn + ffn)
    total = emb + layers

    print(f"  dim {DIM} · layers {N_LAYERS} · heads {N_HEADS} · ffn {FFN_DIM} · vocab {VOCAB:,}")
    print()
    print(f"    embedding table   {emb/1e6:>6.1f}M   {100*emb/total:>3.0f}%   ← just the vocabulary")
    print(f"    {N_LAYERS} layers         {layers/1e6:>6.1f}M   {100*layers/total:>3.0f}%")
    print(f"    {'─'*8}          {'─'*6}")
    print(f"    total             {total/1e6:>6.1f}M")
    print()
    print(f"  Embeddings are tied (input = output), so that table is counted once.")
    print(f"  GPT-3 was 175B. This is about 1/1500th the size, and runs on a laptop.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, choices=[1, 2, 3, 4],
                    help="run one section only")
    args = ap.parse_args()

    sp = Tok(load_tokenizer())

    print("\n\033[1mv11 tokenizer — demos for the TinyStories training video\033[0m")
    print(f"published build · vocab {sp.get_piece_size():,} · pip install v11-tokenizer")

    todo = [args.section] if args.section else [1, 2, 3, 4]
    for n in todo:
        if n == 4:
            section_4()
        else:
            {1: section_1, 2: section_2, 3: section_3}[n](sp)
    print()


if __name__ == "__main__":
    main()
