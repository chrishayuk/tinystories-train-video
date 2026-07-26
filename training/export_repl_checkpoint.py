#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "numpy"]
# ///
"""Bridge a train.py checkpoint into the artefact layout the demos expect.

    export_repl_checkpoint.py --ckpt-dir run_pretrain/ckpt --out model_full.pt

train.py writes `<ckpt>/step_<n>/model.safetensors` + `meta.json`. repl.py,
and train_mathonly.py both load through
`tiny_model_v11.load_from_artifacts(model_v11/, "<name>.pt")`, which wants a
**torch `.pt`** under `model_v11/artifacts/` plus a `model_v11/config.json`.
Different path, different format -- so without this step every demo that
generates text finds nothing.

Deliberately a converter rather than a change to `tiny_model_v11/loader.py`:
that package is vendored from tiny-model/model/v11-core AND is now republished
inside the Hub model repo, so teaching it a second checkpoint format would fork
it in two places at once. Mirrors cell80's own `cn7_export_repl_ckpt.py`.

WHICH SLOT. repl.py's `/full` means "after phase 1/2" and `/compiled` means
"after phase 3, frozen FFN". The 2026-07-25 run is **phase 1 only**, so it
exports to `model_full.pt` and there is deliberately no `model_compiled.pt`:
writing the same weights under both names would put a phase-3 checkpoint that
does not exist into the demo, and Act 1f's whole beat is the difference between
the two.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

ARCH_CONFIG = HERE / "harness_pretrain" / "config.json"
TOKENIZER_JSON = REPO_ROOT / "tokenizer" / "tokenizer.json"
ARTEFACTS = REPO_ROOT / "model_v11"
PUBLISHED_TOKENIZER_SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "run_pretrain" / "ckpt")
    ap.add_argument("--step", type=int, default=None, help="default: highest complete")
    ap.add_argument("--out", default="model_full.pt")
    args = ap.parse_args()
    # Resolved, because the provenance line below records the checkpoint's path
    # relative to the repo root -- and `relative_to` throws outright when one side
    # is relative and the other absolute. A relative --ckpt-dir is the natural
    # thing to type, and it crashed *after* the 600MB load rather than before it.
    args.ckpt_dir = args.ckpt_dir.resolve()

    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    found = {int(m.group(1)): d for d in args.ckpt_dir.iterdir()
             if (m := re.fullmatch(r"step_(\d+)", d.name)) and d.is_dir()}
    ready = {n: d for n, d in found.items() if (d / ".ready").is_file()}
    if not ready:
        sys.exit(f"no complete checkpoint under {args.ckpt_dir} (have {sorted(found)})")
    step = args.step if args.step is not None else max(ready)
    if step not in ready:
        sys.exit(f"step {step} is not complete. Complete: {sorted(ready)}")
    step_dir = ready[step]

    meta = json.loads((step_dir / "meta.json").read_text())
    arch = json.loads(ARCH_CONFIG.read_text())
    tok_sha = sha256_file(TOKENIZER_JSON)

    # Same join publish_pretrain_hf.py enforces, for the same reason: a
    # checkpoint driven by the wrong tokenizer generates fluent nonsense rather
    # than failing, so it has to be checked and not assumed.
    if tok_sha != PUBLISHED_TOKENIZER_SHA256:
        sys.exit(f"\nREFUSING -- {TOKENIZER_JSON} is not the published v11 build.\n"
                 f"  expected {PUBLISHED_TOKENIZER_SHA256}\n  found    {tok_sha}\n")
    if meta.get("tokenizer_hash") != tok_sha:
        sys.exit(f"\nREFUSING -- checkpoint was not trained with this tokenizer.\n"
                 f"  checkpoint tokenizer_hash: {meta.get('tokenizer_hash') or '(absent)'}\n"
                 f"  tokenizer/tokenizer.json : {tok_sha}\n")

    state = load_file(str(step_dir / "model.safetensors"))
    tok_vocab = Tokenizer.from_file(str(TOKENIZER_JSON)).get_vocab_size()
    embed_rows = tuple(state["embed.weight"].shape)[0]
    if not (embed_rows == arch["vocab_size"] == tok_vocab):
        sys.exit(f"\nREFUSING -- vocabulary disagreement.\n"
                 f"  embed.weight rows {embed_rows}\n  config.json {arch['vocab_size']}\n"
                 f"  tokenizer {tok_vocab}\n")

    (ARTEFACTS / "artifacts").mkdir(parents=True, exist_ok=True)
    # config.json is what ModelConfig reads; carry the arch verbatim and record
    # which checkpoint filled this slot, so the dir is self-describing.
    cfg = dict(arch)
    cfg["exported_from"] = {
        "checkpoint": str(step_dir.relative_to(REPO_ROOT)),
        "step": meta.get("step"),
        "tokens": meta.get("tokens"),
        "tokenizer_hash": tok_sha,
        "phase": "phase 1 only -- no frozen-FFN phase 3, so no model_compiled.pt",
    }
    (ARTEFACTS / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    out_path = ARTEFACTS / "artifacts" / args.out
    torch.save(state, str(out_path))
    print(f"exported {step_dir.relative_to(REPO_ROOT)} (step {meta.get('step')}, "
          f"{(meta.get('tokens') or 0)/1e6:.2f}M tokens)")
    print(f"      -> {out_path.relative_to(REPO_ROOT)}")
    print(f"      -> {(ARTEFACTS / 'config.json').relative_to(REPO_ROOT)}")

    # Verify through the path the demos actually use, and generate -- a shape
    # check passes straight through a model that emits garbage.
    from tiny_model_v11 import load_from_artifacts
    model, config = load_from_artifacts(ARTEFACTS, checkpoint=args.out, device="cpu")
    t = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids = [t.token_to_id("<s>")] + t.encode("Once upon a time").ids
    with torch.no_grad():
        for _ in range(30):
            ids.append(int(model(torch.tensor([ids[-config.max_seq:]]))[0, -1].argmax()))
    print(f"  load_from_artifacts OK (vocab {config.vocab_size:,}), greedy sample:")
    print(f"    {' '.join(t.decode(ids).split())[:160]!r}")


if __name__ == "__main__":
    main()
