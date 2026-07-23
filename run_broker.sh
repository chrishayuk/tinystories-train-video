#!/usr/bin/env bash
# The closing shot: model emits a cell call, a Z80 executes it, the model
# narrates the verified number back into the story.
#
#   ./run_broker.sh                      # the 157 ÷ 16 demo, one clean call
#   ./run_broker.sh "Some prose ending in a number slot"
#
# Used twice in SCRIPT.md: the cold-open flash-forward and Act 4d.
#
# Verified working 2026-07-23. Output:
#   [broker] model called safe_div(157, 16) -> cell returned 9
#   OUTPUT: <call> ⟨safe_div⟩ 157 16 </call> 9 sweets. The children smiled.
#
# Two things here are NOT obvious and cost real time to rediscover:
#
#  1. cell80_py exists only in the uv cache — there is no installable pin — and
#     MOST of the cached builds are too old: they reject the `a / b` in
#     safe_div.rs with "unsupported statement expression". CELL80_PY below is a
#     build that compiles it. If it ever disappears, find another with:
#
#       for d in $(find ~/.cache/uv/archive-v0 -maxdepth 2 -name cell80_py -type d); do
#         p=$(dirname $d)
#         PYTHONPATH=$p uv run --quiet python -c "
#       import cell80_py
#       h = cell80_py.CellHost()
#       h.add_source('safe_div', open('$CELLS/safe-arith/safe_div.rs').read())
#       print('COMPILES', '$p')" 2>/dev/null
#       done
#
#  2. --max-tokens 21 --max-calls 1 is deliberate. The default 60 loops the
#     same call and narration three times, which reads as a bug on camera.
#     21 lands exactly on "The children smiled."

set -euo pipefail

CN7=~/chris-source/cell80/experiments/cell-native-architectures
CELLS=~/chris-source/cell80/cell80/cells
V11_CORE=~/chris-source/tiny-model/model/v11-core
CELL80_PY=~/.cache/uv/archive-v0/nL4y8VCxGig7bi9v937SL

PROMPT="${1:-157 sweets were shared fairly between 16 children. The sharing machine said each child gets}"

for p in "$CN7" "$V11_CORE" "$CELL80_PY"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

cd "$CN7"
PYTHONPATH="$CELL80_PY" uv run --quiet \
  --with torch --with sentencepiece --with numpy \
  --with-editable "$V11_CORE" \
  python cn7_broker.py \
    --ckpt cn7_ckpt_midtrain.pt \
    --max-calls 1 \
    --max-tokens 21 \
    --prompt "$PROMPT"
