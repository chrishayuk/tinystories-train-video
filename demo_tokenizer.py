#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers>=0.20", "sentencepiece>=0.2"]
# ///
"""On-camera tokenizer demos for the TinyStories training video.

Self-contained: reads only from ./tokenizer/, no repo paths, no network.

  uv run demo_tokenizer.py

Sections map to the script:
  1  Act 1b  -- what a token is
  2  Act 2b  -- v11 is a knowledge-first tokenizer
  3  Act 2b  -- digits split, number words don't  (why maths is unnatural here)
  4  Act 1a  -- a third of the model is its vocabulary
  5  Act 2c  -- the tokenizer that wasn't: same pieces, different IDs

Run with --section N to show just one on camera.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE = HERE / "tokenizer" / "v11_native.model"        # what v11 was TRAINED with
COMMITTED = HERE / "tokenizer" / "tokenizer_committed.json"  # what sits in the repo

# v11 architecture (model/v11/config.json)
DIM, N_LAYERS, FFN_DIM, N_HEADS, N_KV_HEADS, VOCAB = 512, 20, 2048, 8, 4, 71261


def rule(title):
    print(f"\n\033[1m{title}\033[0m")
    print("─" * len(title))


def load_native():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(NATIVE))
    return sp


def load_committed():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(COMMITTED))


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
BLOCKS = [
    (0, "special tokens"),
    (4, "the ten digits — right at the front, before anything else"),
    (120, "punctuation"),
    (182, "Greek letters, bare and space-prefixed"),
    (300, "maths and set operators"),
    (1000, "tree-sitter AST node types — the grammars of 77 languages"),
    (8000, "more parser node types"),
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
    print(f"    {n-256:>6}  {' '.join(sp.id_to_piece(i) for i in range(n-256, n-248))}")
    print("            \033[2mbyte fallback — so nothing is ever unrepresentable\033[0m")
    print()
    print("  That is not a statistical artefact. That is a hand-built index.")
    print()
    print("  Design principle, from the v11 README, written long before any")
    print("  experiment in this video existed:")
    print("      'Every token is a potential compilation target.'")
    print()
    print("  Note where the digits landed: IDs 4-13, ahead of all language.")


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

    print("\n  So the same quantity is one token as a word, three as digits:")
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


# ── 5 ── Act 2c: the tokenizer that wasn't ───────────────────────────────────
def section_5(sp):
    rule("5. The tokenizer that wasn't")
    hf = load_committed()

    print("  Two files. One is what the checkpoint was actually trained with;")
    print("  the other is what was sitting next to it in the repo.")
    print()
    print(f"    native v11.model      vocab {sp.get_piece_size():,}")
    print(f"    committed .json       vocab {hf.get_vocab_size():,}")
    print()
    print("  The pieces are IDENTICAL. The text splits exactly the same way:")
    print()

    for text in ["Once upon a time", "three times four is twelve"]:
        a_p, b_p = pieces(sp, text), hf.encode(text).tokens
        a_i, b_i = sp.encode(text), hf.encode(text).ids
        print(f'    "{text}"')
        print(f"      native    {a_p}")
        print(f"      committed {b_p}")
        print(f"      pieces identical: {a_p == b_p}")
        print(f"      native    ids  {a_i}")
        print(f"      committed ids  {b_i}")
        print(f"      \033[1mIDs identical: {a_i == b_i}\033[0m")
        print()

    print("  Same words. Same splits. Every ID points at a different piece.")
    print()
    print("  Nothing looks wrong. Encoding works, decoding works, the pieces are")
    print("  right. But feed those IDs to the checkpoint and it scores ~18 nats")
    print("  on ordinary English — where uniform random guessing scores ~11.")
    print()
    print("  A model cannot be worse than random by accident. Being reliably")
    print("  wrong takes information. Through the native mapping: 0.66.")
    print()
    print("  An earlier experiment in this programme ran through the wrong")
    print("  mapping and nobody noticed — because a broken setup still produces")
    print("  numbers. It still produces graphs. It still produces conclusions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, choices=[1, 2, 3, 4, 5],
                    help="run one section only")
    args = ap.parse_args()

    for path, what in ((NATIVE, "native v11.model"), (COMMITTED, "committed tokenizer.json")):
        if not path.exists():
            sys.exit(f"missing {what}: {path}")

    sp = load_native()

    print("\n\033[1mv11 tokenizer — demos for the TinyStories training video\033[0m")
    print("using the NATIVE v11.model (the mapping the checkpoint was trained with)")

    todo = [args.section] if args.section else [1, 2, 3, 4, 5]
    for n in todo:
        if n == 4:
            section_4()
        else:
            {1: section_1, 2: section_2, 3: section_3, 5: section_5}[n](sp)
    print()


if __name__ == "__main__":
    main()
