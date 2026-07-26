#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "numpy"]
# ///
"""Act 4d — the model emits a cell call, a Z80 executes it, the model narrates on.

The closing shot of the video, closed for the first time against THIS model rather
than CN-7's. `run_broker.sh` still drives the retired checkpoint through
sentencepiece and cell80's own model code; this drives
`runs/cells-s80/ckpt/step_12504` through the published v11 tokenizer and
`tiny_model_v11`, which is the lineage every other act uses.

The loop is CN-7's -- generate greedily; when `</call>` lands, scan back to
`<call>`, read the cell token and the digit arguments, run the cell, splice the
verified digits in, keep generating. What differs is only which model, which
tokenizer and which token map.

The call format is the one build_cells_corpus.py's S2 rows teach:

    <call> ⟨cell⟩ arg arg </call> RESULT tail
                                  ^^^^^^ injected at ZERO gradient during training,
                                         so the weights never learned it -- and at
                                         inference there is nothing there to say.
                                         That hole is what this fills.

    PYTHONPATH=<cell80_py build> uv run training/broker.py \
        --prompt "840 sweets were shared fairly between 7 children. The sharing machine said each child gets"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

ARTEFACTS = HERE.parent / "model_v11"
TOKEN_MAP = HERE / "data" / "cells" / "cells_token_map.json"
CELLS_DIR = Path.home() / "chris-source" / "cell80" / "cell80" / "cells"
DEFAULT_CKPT = HERE.parent / "runs" / "cells-s80" / "ckpt" / "step_12504"

DIM, GREEN, CYAN, BOLD, RESET = (
    "\033[2m", "\033[92m", "\033[96m", "\033[1m", "\033[0m")


def load_cells_model(ckpt_dir: Path, extended: int):
    """Build at the base vocabulary, grow the tied embedding, then load.

    The same three steps train_cells.py takes on the worker, and the only way this
    checkpoint loads at all: `model_v11/config.json` pins vocab 71,260 for every
    checkpoint in the artefact dir, and this is the one model with 72,052 rows.
    """
    from safetensors.torch import load_file
    from tiny_model_v11 import load_from_artifacts
    from train_cells import resize_embedding

    model, cfg = load_from_artifacts(str(ARTEFACTS), checkpoint="model_full.pt",
                                     device="cpu")
    resize_embedding(model, extended)
    state = load_file(str(ckpt_dir / "model.safetensors"))
    rows = tuple(state["embed.weight"].shape)[0]
    if rows != extended:
        raise SystemExit(f"\nREFUSING: checkpoint has {rows} embedding rows, the "
                         f"token map describes {extended}.\n")
    model.load_state_dict(state, strict=True)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), cfg, device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default="157 sweets were shared fairly between 16 "
                                        "children. The sharing machine said each child gets")
    # 21, for the same reason run_broker.sh pins 21: the default 60 loops the call
    # and the narration several times over, which reads as a bug on camera.
    ap.add_argument("--max-tokens", type=int, default=21)
    ap.add_argument("--max-calls", type=int, default=1)
    args = ap.parse_args()

    try:
        import cell80_py
    except ImportError:
        raise SystemExit(
            "\ncell80_py is not importable. It lives only in the uv cache -- there\n"
            "is no installable pin. Run this through run_broker_v11.sh, which sets\n"
            "PYTHONPATH to a build that compiles the cell sources.\n")

    tmap = json.loads(TOKEN_MAP.read_text())
    call_id, close_id = tmap["call"], tmap["close"]
    inv = {v: k for k, v in tmap["cells"].items()}

    model, cfg, device = load_cells_model(args.ckpt_dir, tmap["extended_vocab"])
    from demo_common import V11Tokenizer
    tok = V11Tokenizer()
    print(f"{DIM}cells checkpoint {args.ckpt_dir.name} on {device} · "
          f"vocab {tmap['base_vocab']} -> {tmap['extended_vocab']}{RESET}\n")

    ids = tok.encode(args.prompt)
    if tok.bos_id() >= 0:
        ids = [tok.bos_id()] + ids
    emitted: list[int] = []
    host, handles, calls = cell80_py.CellHost(), {}, 0

    print(f"{BOLD}PROMPT:{RESET} {args.prompt}")
    with torch.no_grad():
        while len(emitted) < args.max_tokens:
            logits = model(torch.tensor([(ids + emitted)[-cfg.max_seq:]],
                                        device=device))[0, -1]
            t = int(logits.argmax())
            if t == tok.eos_id():
                break
            emitted.append(t)
            if t != close_id or calls >= args.max_calls:
                continue

            # Scan back to the <call> that this </call> closes.
            try:
                back = next(i for i, x in enumerate(reversed(emitted)) if x == call_id)
            except StopIteration:
                continue
            span = emitted[len(emitted) - 1 - back:]
            cell_toks = [x for x in span if x in inv]
            digits = [x for x in span if x < tmap["base_vocab"]]
            if not cell_toks:
                continue
            name = inv[cell_toks[0]]
            cargs = [int(s) for s in tok.decode(digits).split()
                     if s.lstrip("-").isdigit()]

            if name not in handles:
                src = next(CELLS_DIR.rglob(f"{name}.rs"), None)
                if src is None:
                    raise SystemExit(f"\nno cell source found for {name!r} under "
                                     f"{CELLS_DIR}\n")
                host.add_source(name, src.read_text())
                handles[name] = host.load(name)
            r = host.run(handles[name], cargs)
            res = r["result"] if r.get("halt") == "returned" else None
            print(f"  {CYAN}[cell]{RESET} {name}({', '.join(map(str, cargs))}) "
                  f"-> {BOLD}{res}{RESET}   {DIM}executed on the Z80{RESET}")
            if res is not None:
                emitted.extend(tok.encode(f" {res}"))
            calls += 1

    # Render, colouring the two sources differently -- the viewer has to be able to
    # see at a glance which characters the model produced and which the runtime did.
    # If the answer looks like model output, the act has no point.
    out, run = [], []
    for t in emitted:
        marker = ("<call>" if t == call_id else "</call>" if t == close_id
                  else f"⟨{inv[t]}⟩" if t in inv else None)
        if marker is None:
            run.append(t)
        else:
            if run:
                out.append(tok.decode(run))
                run = []
            out.append(f"{CYAN}{marker}{RESET}")
    if run:
        out.append(tok.decode(run))
    print(f"{BOLD}OUTPUT:{RESET} {' '.join(out).strip()}")


if __name__ == "__main__":
    main()
