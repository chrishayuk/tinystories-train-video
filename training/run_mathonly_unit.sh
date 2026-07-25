#!/bin/sh
# chuk-train entrypoint for the Act 3 maths midtrain.
#
# Unlike the pretrain unit, this one has two inputs that are NOT in the repo --
# both gitignored, so neither is in the code-unit tar:
#
#   1. the base checkpoint (606MB) -> pulled from the Hub. It is published, so the
#      worker fetches the exact bytes the local runs used; the sha is checked
#      against the card's identity rather than trusted.
#   2. the corpus -> rebuilt here. It is deterministic given --seed 90, so
#      rebuilding is equivalent to shipping it and avoids a ~100MB upload. Costs a
#      TinyStories replay stream from the pinned HF revision.
#
# Deliberately does NOT install torch: the worker image ships a CUDA-matched build
# and `pip install torch` risks replacing it with a mismatched or CPU wheel.
set -e
cd "$(dirname "$0")/.."

python3 -m pip install --quiet "tokenizers>=0.20" "datasets>=2.18" \
    "safetensors>=0.4" "huggingface_hub>=0.24"

# export, not plain assignment: the heredoc below is quoted (<<'PY') so nothing is
# shell-expanded inside it and the Python reads these from the environment. Without
# export they are shell-local and the child dies on KeyError.
export BASE_REPO=chrishayuk/v11-tinystories-115m-base
export EXPECT_SHA=1841e0581574629716b646dacd4e70feaca153a8adc5ecb0b77e0e2ebdf78d9c

echo "[mathonly] fetching base from $BASE_REPO"
python3 - <<'PY'
import hashlib, json, os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
repo = os.environ.get("BASE_REPO", "chrishayuk/v11-tinystories-115m-base")
expect = os.environ["EXPECT_SHA"]
art = pathlib.Path("model_v11/artifacts"); art.mkdir(parents=True, exist_ok=True)

st = hf_hub_download(repo, "model.safetensors")
got = hashlib.sha256(pathlib.Path(st).read_bytes()).hexdigest()
if got != expect:
    sys.exit(f"REFUSING: base sha {got} != published identity {expect}")
print(f"[mathonly] base sha verified {got[:16]}...")

# the demos and train_mathonly.py load a torch .pt via load_from_artifacts
import torch
from safetensors.torch import load_file
torch.save(load_file(st), art / "model_full.pt")
shutil.copyfile(hf_hub_download(repo, "config.json"), "model_v11/config.json")
PY

echo "[mathonly] building corpus (deterministic, seed 90)"
python3 training/build_mathonly_corpus.py --drill 90000 --seed 90

echo "[mathonly] training"
exec python3 training/train_mathonly.py \
    --tokens "${MATHONLY_TOKENS:-12000000}" --bs 16 --lr 1e-4 --warmup 200 \
    --seed "${CHUK_SEED:-80}" --val-every 2000 --sample-every 250 --save-every 1500
