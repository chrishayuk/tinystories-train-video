#!/bin/sh
# chuk-train entrypoint for phase 3 — freeze the FFN, retrain attention.
#
# Simpler than the two midtrain arms in one way that matters: there is no corpus
# to fetch or verify. Phase 3 trains on plain TinyStories at the pinned revision,
# read straight from the parquet shards over HTTP range requests, so the only
# input with an identity to check is the base checkpoint.
set -e
cd "$(dirname "$0")/.."

# Same allocator setting, same reason as the other two arms: fragmentation, not
# capacity, is what OOMs a 16GB card here.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m pip install --quiet "tokenizers>=0.20" "safetensors>=0.4" \
    "huggingface_hub>=0.24" "pyarrow>=14"

export BASE_REPO=chrishayuk/v11-tinystories-115m-base
export EXPECT_SHA=1841e0581574629716b646dacd4e70feaca153a8adc5ecb0b77e0e2ebdf78d9c

echo "[phase3] fetching base from $BASE_REPO"
python3 - <<'PY'
import hashlib, os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
repo, expect = os.environ["BASE_REPO"], os.environ["EXPECT_SHA"]
art = pathlib.Path("model_v11/artifacts"); art.mkdir(parents=True, exist_ok=True)
st = hf_hub_download(repo, "model.safetensors")
got = hashlib.sha256(pathlib.Path(st).read_bytes()).hexdigest()
if got != expect:
    sys.exit(f"REFUSING: base sha {got} != published identity {expect}")
print(f"[phase3] base sha verified {got[:16]}...")
import torch
from safetensors.torch import load_file
torch.save(load_file(st), art / "model_full.pt")
shutil.copyfile(hf_hub_download(repo, "config.json"), "model_v11/config.json")
PY

echo "[phase3] training"
exec python3 training/train_phase3.py \
    --tokens "${PHASE3_TOKENS:-8000000}" --bs 4 --lr 1.5e-4 --warmup 200 \
    --seed "${CHUK_SEED:-43}" --sample-every 500 --save-every 1500
