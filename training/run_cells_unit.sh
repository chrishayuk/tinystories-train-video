#!/bin/sh
# chuk-train entrypoint for the CN-9 cells midtrain (Act 4's arm).
#
# Differs from the maths-only entrypoint in ONE structural way, and it is not a
# preference: that corpus is rebuilt on the worker because it is deterministic
# given a seed, and rebuilding beats a 100MB upload. THIS corpus cannot be
# rebuilt here. Every answer in it is signed by executing a real cell through
# `cell80_py`, a compiled extension plus 790 cell sources that no training
# worker has or should have.
#
# So it is fetched and sha-verified, exactly as the base checkpoint is. The
# cell80 dependency stays on the build side, where it belongs, and the worker
# handles bytes with an identity rather than a toolchain.
set -e
cd "$(dirname "$0")/.."

# Same allocator setting, same reason as the maths arm: fragmentation, not
# capacity, is what OOMs a 16GB card here. The extended vocabulary makes the
# logits tensor ~1.1% larger, which changes nothing about that argument.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m pip install --quiet "tokenizers>=0.20" "datasets>=2.18" \
    "safetensors>=0.4" "huggingface_hub>=0.24"

export BASE_REPO=chrishayuk/v11-tinystories-115m-base
export EXPECT_SHA=1841e0581574629716b646dacd4e70feaca153a8adc5ecb0b77e0e2ebdf78d9c

echo "[cells] fetching base from $BASE_REPO"
python3 - <<'PY'
import hashlib, os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
repo, expect = os.environ["BASE_REPO"], os.environ["EXPECT_SHA"]
art = pathlib.Path("model_v11/artifacts"); art.mkdir(parents=True, exist_ok=True)
st = hf_hub_download(repo, "model.safetensors")
got = hashlib.sha256(pathlib.Path(st).read_bytes()).hexdigest()
if got != expect:
    sys.exit(f"REFUSING: base sha {got} != published identity {expect}")
print(f"[cells] base sha verified {got[:16]}...")
import torch
from safetensors.torch import load_file
torch.save(load_file(st), art / "model_full.pt")
shutil.copyfile(hf_hub_download(repo, "config.json"), "model_v11/config.json")
PY

# The corpus and its token map, fetched together. The map is what says which id
# is <call> and which cell owns which row of the extended embedding; a corpus
# paired with the wrong map trains happily and means something else, so they
# travel as a pair and the trainer cross-checks them.
export CELLS_REPO="${CELLS_REPO:-chrishayuk/v11-cells-midtrain-corpus}"

# The identity these bytes MUST hash to, pinned here the same way the maths arm
# pins its corpus. Without it the fetch is "whatever that repo happens to serve
# today", which is exactly the property content-addressing exists to remove --
# and a corpus is the one input a training run cannot sanity-check by reading.
# Override with CELLS_EXPECT_SHA= (empty) to run against unverified bytes.
: "${CELLS_EXPECT_SHA=2115d6aeff3428e217ef2903a8030facd511dcb00183e9fc3faaf49d01038767}"
export CELLS_EXPECT_SHA

echo "[cells] fetching corpus from $CELLS_REPO"
python3 - <<'PY'
import hashlib, os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
repo = os.environ["CELLS_REPO"]
# Its own root, because the identity below is computed over a DIRECTORY.
data = pathlib.Path("training/data/cells"); data.mkdir(parents=True, exist_ok=True)
for name in ("cells_corpus.jsonl", "cells_token_map.json"):
    p = hf_hub_download(repo, name, repo_type="dataset")
    shutil.copyfile(p, data / name)
    print(f"[cells] fetched {name} ({(data / name).stat().st_size/1e6:.1f} MB)")

# The chuk-datasets identity, recomputed over what actually landed -- the same
# function build_cells_corpus.py prints, so a mismatch here means the bytes on
# this worker are not the bytes the corpus was registered as.
expect = os.environ.get("CELLS_EXPECT_SHA", "")
if expect:
    files = sorted((p for p in data.rglob("*") if p.is_file()),
                   key=lambda p: str(p.relative_to(data)))
    shards, off = [], 0
    for f in files:
        b = f.read_bytes()
        shards.append({"sha256": hashlib.sha256(b).hexdigest(),
                       "size": str(len(b)), "offset": str(off)})
        off += len(b)
    import json
    core = {"schema": "chuk-manifest-core-1", "shards": shards}
    jcs = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    got = hashlib.sha256(jcs.encode()).hexdigest()
    if got != expect:
        sys.exit(f"REFUSING: corpus identity {got} != expected {expect}")
    print(f"[cells] corpus identity verified {got[:16]}...")
else:
    print("[cells] WARNING: CELLS_EXPECT_SHA unset -- training on unverified corpus bytes")
PY

echo "[cells] training"
exec python3 training/train_cells.py \
    --tokens "${CELLS_TOKENS:-12000000}" --bs 16 --lr 1e-4 --warmup 200 \
    --seed "${CHUK_SEED:-80}" --val-every 2000 --sample-every 250 --save-every 1500
