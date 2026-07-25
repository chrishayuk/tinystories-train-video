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

# Fragmentation, not capacity, is what OOMs this on a 16GB card. The run needs
# ~2GB live (115M weights + grads + AdamW state) but its allocator high-water
# mark reached 13.84 of 16.11 GB and stayed: training batches of 16 replay rows
# at 256 tokens are 4,096 positions = 1.17GB of logits held through the backward,
# and val_nll every 2,000 steps plus sample() every 250 churn allocations of
# different shapes on top. Classic fragmentation, and the exact case
# expandable_segments exists for -- torch itself suggests it in the OOM message.
#
# Deliberately chosen over lowering --bs or capping batch positions: both of
# those change batch composition, which changes gradients, which changes the
# experiment. This changes only how the allocator lays memory out.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# The chuk-datasets identity this corpus MUST hash to. Every worker rebuilds it,
# so without this a rebuild that silently differs trains happily on the wrong bytes
# -- which is how two runs ended up validating on different data (663 rows vs 710)
# and having their val-NLLs compared anyway.
#
# With it, each worker re-proves determinism before spending any GPU time, and a
# multi-seed replicate is guaranteed to share a corpus rather than assumed to.
# Override with MATHONLY_EXPECT_SHA= (empty) to build without checking.
#
# Exported, not shell-local: train_mathonly.py stamps it into every checkpoint's
# meta.json. A midtrain checkpoint inherits its corpus rather than producing it,
# so without this the bytes carry no record of WHICH corpus taught them -- and
# that is precisely the join two runs need to be comparable at all.
: "${MATHONLY_EXPECT_SHA=ff7bf26b359914344317729678884fb9fd8f1bac8e6916d67de416ab46fdf33f}"
export MATHONLY_EXPECT_SHA

echo "[mathonly] building corpus (seed 90; identity-checked)"
python3 training/build_mathonly_corpus.py --drill 90000 --seed 90 \
    ${MATHONLY_EXPECT_SHA:+--expect-sha "$MATHONLY_EXPECT_SHA"}

echo "[mathonly] training"
exec python3 training/train_mathonly.py \
    --tokens "${MATHONLY_TOKENS:-12000000}" --bs 16 --lr 1e-4 --warmup 200 \
    --seed "${CHUK_SEED:-80}" --val-every 2000 --sample-every 250 --save-every 1500
