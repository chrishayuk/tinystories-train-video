#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "safetensors>=0.4", "huggingface_hub>=0.24", "numpy"]
# ///
"""Fetch the published base model and maths corpus into the layout everything here expects.

    uv run training/fetch_published.py             # both
    uv run training/fetch_published.py --corpus    # just the corpus

This is step one of reproducing anything in this repo. Both artifacts are
verified against their published identities before they are put in place, so a
truncated download or a repo that has moved underneath you fails here rather than
several hours into a training run.

    model_v11/config.json                     architecture, from the model repo
    model_v11/artifacts/model_full.pt         the 16M-token base, as a torch .pt
    training/data/mathonly_corpus.jsonl       the mid-train corpus
    training/data/mathonly_held_out.json      the facts deliberately withheld

WHY THE CORPUS IS FETCHED FILE BY FILE rather than with `hf download --local-dir`:
its identity is a chuk-datasets content_sha computed over the DIRECTORY, so the
corpus root has to hold those two files and nothing else. `--local-dir` writes a
`.cache/` tree of download metadata beside them even with `--include`, and a root
with that in it hashes to `a9d67ca6…` instead of `ff7bf26b…`. Nothing errors --
training runs happily either way -- so the wrong identity would only be found by
whatever recomputes it later, which is exactly the class of silent failure
content-addressing exists to remove.

WHY THE BASE IS CONVERTED. train.py and the Hub repo hold `model.safetensors`;
repl.py and both trainers load through `tiny_model_v11.load_from_artifacts`, which
wants a torch `.pt` under `model_v11/artifacts/`. Same weights, different
container. Written to `model_full.pt` and deliberately NOT to `model_compiled.pt`:
that slot means "after phase 3", which has not been run on this lineage, and
filling it with these weights would put a checkpoint that does not exist into the
demo.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

BASE_REPO = "chrishayuk/v11-tinystories-115m-base"
BASE_SHA = "1841e0581574629716b646dacd4e70feaca153a8adc5ecb0b77e0e2ebdf78d9c"

CORPUS_REPO = "chrishayuk/v11-mathonly-midtrain-corpus"
CORPUS_SHA = "ff7bf26b359914344317729678884fb9fd8f1bac8e6916d67de416ab46fdf33f"
CORPUS_FILES = ("mathonly_corpus.jsonl", "mathonly_held_out.json")


def fetch_base() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"base   {BASE_REPO}")
    st = Path(hf_hub_download(BASE_REPO, "model.safetensors"))
    got = hashlib.sha256(st.read_bytes()).hexdigest()
    if got != BASE_SHA:
        sys.exit(f"\nREFUSING -- base model sha256 {got}\n"
                 f"           is not the published identity {BASE_SHA}\n")
    print(f"       sha256 verified {got[:16]}… ({st.stat().st_size/1e6:.0f} MB)")

    art = REPO_ROOT / "model_v11" / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    torch.save(load_file(str(st)), art / "model_full.pt")
    shutil.copyfile(hf_hub_download(BASE_REPO, "config.json"),
                    REPO_ROOT / "model_v11" / "config.json")
    print(f"       -> model_v11/artifacts/model_full.pt + model_v11/config.json")


def fetch_corpus() -> None:
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, str(HERE))
    from build_mathonly_corpus import content_sha

    print(f"corpus {CORPUS_REPO}")
    data = REPO_ROOT / "training" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for n in CORPUS_FILES:
        shutil.copyfile(hf_hub_download(CORPUS_REPO, n, repo_type="dataset"), data / n)
        print(f"       {n} ({(data / n).stat().st_size/1e6:.1f} MB)")

    # Hashed in a clean staging copy, not in place: training/data/ is also where
    # shared_val_710.jsonl and the cells corpus land, and the identity is over the
    # directory. Checking in place would fail for everyone who has both arms.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for n in CORPUS_FILES:
            shutil.copyfile(data / n, stage / n)
        got = content_sha(stage)
    if got != CORPUS_SHA:
        sys.exit(f"\nREFUSING -- corpus identity {got}\n"
                 f"           is not the registered {CORPUS_SHA}\n")
    print(f"       content_sha verified {got[:16]}…")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", action="store_true", help="only the base model")
    ap.add_argument("--corpus", action="store_true", help="only the maths corpus")
    args = ap.parse_args()
    both = not (args.base or args.corpus)

    if both or args.base:
        fetch_base()
    if both or args.corpus:
        fetch_corpus()

    print("\nready. Next:")
    print("  uv run repl.py                       talk to the base model")
    print(f"  export MATHONLY_EXPECT_SHA={CORPUS_SHA}")
    print("  uv run training/train_mathonly.py --tokens 12000000 --bs 16 --lr 1e-4 \\")
    print("      --warmup 200 --seed 80 --val-every 2000 --sample-every 250 --save-every 1500")


if __name__ == "__main__":
    main()
