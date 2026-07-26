#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "numpy"]
# ///
"""Load the cells checkpoint and ask it the Act 4 questions.

WHY THIS EXISTS, AND WHY export_repl_checkpoint.py CANNOT DO IT. The cells arm is
the one model in this project with a different vocabulary -- 72,052 rows against
the published tokenizer's 71,260, which is the whole reason it needed a resize.
But `model_v11/config.json` is a SINGLE file shared by every checkpoint in
`model_v11/artifacts/`, and `load_from_artifacts` builds the model from it:

    TinyModel(vocab_size=config.vocab_size, ...)   # 71,260, always

So a 72,052-row state dict cannot be loaded through that path at all, and
export_repl_checkpoint.py refuses it -- correctly, because from where it stands
a vocabulary disagreement is exactly the failure it exists to catch.

The resolution is the one train_cells.py already uses on the worker: build at the
base vocabulary, grow the tied embedding, THEN load. The extension is declared
rather than inferred, which is what makes it safe to allow here and right to
refuse in the general exporter.

    uv run training/load_cells_checkpoint.py runs/cells-s80/ckpt/step_12504
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

ARTEFACTS = HERE.parent / "model_v11"
TOKEN_MAP = HERE / "data" / "cells" / "cells_token_map.json"

# The five Act 2a/3 prompts, then the two only this arm can answer.
PROMPTS = [
    ("7 + 5 =", "in-range canonical (12)"),
    ("394 + 251 =", "one digit past the range (645)"),
    ("Lily had 3 apples. Tom gave Lily 4 more. Now Lily has", "in-range, digits (7)"),
    ("Once upon a time", "storytelling — watch for forgetting"),
]
CELL_PROMPTS = [
    ("The truck brought 47 crates with 63 apples in each crate. "
     "The counting machine worked it out:", "beyond-tier: should CALL, not answer"),
    ("840 sweets were shared fairly between 7 children. "
     "The sharing machine said each child gets", "beyond-tier division: should CALL"),
]


@torch.no_grad()
def generate(model, tok, ids, max_new, max_seq):
    out = []
    for _ in range(max_new):
        nxt = int(model(torch.tensor([ids[-max_seq:]], device=model.embed.weight.device))[0, -1].argmax())
        ids.append(nxt)
        out.append(nxt)
        if nxt == tok.eos_id():
            break
    return out


def main() -> None:
    ckpt_dir = Path(sys.argv[1] if len(sys.argv) > 1
                    else "runs/cells-s80/ckpt/step_12504").resolve()
    from safetensors.torch import load_file
    from tiny_model_v11 import load_from_artifacts
    from train_cells import resize_embedding
    from demo_common import V11Tokenizer

    tmap = json.loads(TOKEN_MAP.read_text())
    extended = tmap["extended_vocab"]

    meta = json.loads((ckpt_dir / "meta.json").read_text())
    model, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint="model_full.pt",
                                     device="cpu")
    n_old = resize_embedding(model, extended)
    state = load_file(str(ckpt_dir / "model.safetensors"))
    rows = tuple(state["embed.weight"].shape)[0]
    if rows != extended:
        raise SystemExit(f"\nREFUSING: checkpoint has {rows} embedding rows, "
                         f"the token map describes {extended}.\n")
    model.load_state_dict(state, strict=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"== cells checkpoint step {meta.get('step')} on {device} | "
          f"vocab {n_old} -> {extended} ==")

    tok = V11Tokenizer()
    call_id = tmap["call"]
    inv = {v: k for k, v in tmap["cells"].items()}

    for prompt, note in PROMPTS:
        ids = tok.encode(prompt)
        if tok.bos_id() >= 0:
            ids = [tok.bos_id()] + ids
        new = generate(model, tok, ids, 12, cfg.max_seq)
        print(f"  {prompt!r}\n    -> {tok.decode(ids)[len(tok.decode(ids[:-len(new)])):]!r}"
              f"\n       {note}")

    print("\n  does it delegate?")
    for prompt, note in CELL_PROMPTS:
        ids = tok.encode(prompt)
        if tok.bos_id() >= 0:
            ids = [tok.bos_id()] + ids
        new = generate(model, tok, ids, 12, cfg.max_seq)
        pieces = ["<call>" if n == call_id else f"⟨{inv[n]}⟩" if n in inv
                  else tok.id_to_piece(n) for n in new]
        called = any(p == "<call>" for p in pieces)
        print(f"  {prompt[:58]!r}…\n    -> {''.join(pieces)}   "
              f"{'CALLED' if called else 'NO CALL'}\n       {note}")


if __name__ == "__main__":
    main()
