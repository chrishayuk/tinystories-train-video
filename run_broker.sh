#!/usr/bin/env bash
# The closing shot: model emits a cell call, a Z80 executes it, the model
# narrates the verified number back into the story.
#
#   ./run_broker.sh                      # the 157 / 16 demo, one clean call
#   ./run_broker.sh "Some prose ending in a number slot"
#
# Used twice in SCRIPT.md: the cold-open flash-forward and Act 4d.
#
# Verified working 2026-07-26 against the CURRENT model. Output:
#   [cell] safe_div(157, 16) -> 9   executed on the Z80
#   OUTPUT: <call> ⟨safe_div⟩ 157 16 </call> 9 sweets. The children smiled.
#
# ---------------------------------------------------------------------------
# REWRITTEN 2026-07-26. This used to drive cell80's `cn7_broker.py` against
# `cn7_ckpt_midtrain.pt` -- the RETIRED checkpoint, through sentencepiece and
# cell80's own model code. That model cannot be driven by the published v11
# tokenizer at all, so the "verified working 2026-07-23" note it carried was
# true of a model no longer in this video.
#
# It now runs training/broker.py against runs/cells-s80/ckpt/step_12504: the
# published tokenizer, tiny_model_v11, and the checkpoint the audience watches
# being trained in Act 4b. One model lineage, start to finish.
#
# Two things here are NOT obvious and cost real time to rediscover:
#
#  1. cell80_py exists only in the uv cache -- there is no installable pin --
#     and MOST cached builds are too old: they reject the `a / b` in
#     safe_div.rs with "unsupported statement expression". CELL80_PY below is
#     a build that compiles it. If it ever disappears, find another with:
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
#  2. --max-tokens 26, and the reason CHANGED. The old note here said 60 loops
#     the call and narration three times and pinned 21 to land on "The children
#     smiled." That trap is GONE on this model -- 60 now stops in the same
#     place, because the corpus EOS fix taught it to stop. 26 is simply the
#     shortest length that reaches the full sentence through this tokenizer;
#     21 cuts it off at "The children".
# ---------------------------------------------------------------------------

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CELLS=~/chris-source/cell80/cell80/cells
CELL80_PY=~/.cache/uv/archive-v0/nL4y8VCxGig7bi9v937SL
CKPT="$HERE/runs/cells-s80/ckpt/step_12504"

PROMPT="${1:-157 sweets were shared fairly between 16 children. The sharing machine said each child gets}"

for p in "$CELLS" "$CELL80_PY" "$CKPT"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

PYTHONPATH="$CELL80_PY" uv run --quiet "$HERE/training/broker.py" \
  --ckpt-dir "$CKPT" \
  --max-calls 1 \
  --max-tokens 26 \
  --prompt "$PROMPT"
