#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Act 1a — where the parameters actually go.

  uv run show_params.py

The tokenizer demos moved to the published CLI (`cargo install v11-cli`, then
`v11 vocab --blocks`), which is a better thing to put on camera: viewers can run
the identical command. This one stayed behind because it isn't a tokenizer
question at all -- it's model arithmetic that happens to be *about* the
vocabulary.

Reads training/harness_pretrain/config.json rather than hardcoding the
architecture, so it cannot drift from the config the Act 1e run actually uses.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "training" / "harness_pretrain" / "config.json"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def main(argv=None) -> None:
    cfg = json.loads(CONFIG.read_text())
    dim, layers_n = cfg["dim"], cfg["n_layers"]
    heads, kv_heads = cfg["n_heads"], cfg["n_kv_heads"]
    ffn_dim, vocab = cfg["ffn_dim"], cfg["vocab_size"]
    head_dim = dim // heads

    emb = vocab * dim
    attn = dim * dim * 2 + dim * kv_heads * head_dim * 2
    ffn = 3 * dim * ffn_dim
    layers = layers_n * (attn + ffn)
    total = emb + layers

    print(f"\n{BOLD}A third of the model is its vocabulary{RESET}")
    print("─" * 38)
    print(f"  dim {dim} · layers {layers_n} · heads {heads} · ffn {ffn_dim} · vocab {vocab:,}")
    print()
    print(f"    embedding table   {emb/1e6:>6.1f}M   {100*emb/total:>3.0f}%   ← just the vocabulary")
    print(f"    {layers_n} layers         {layers/1e6:>6.1f}M   {100*layers/total:>3.0f}%")
    print(f"    {'─'*8}          {'─'*6}")
    print(f"    total             {total/1e6:>6.1f}M")
    print()
    print(f"{DIM}  Embeddings are tied (input = output), so that table is counted once.")
    print(f"  GPT-3 was 175B. This is about 1/1500th the size, and runs on a laptop.{RESET}\n")


if __name__ == "__main__":
    main()
