#!/bin/sh
# v11-pretrain entrypoint. Deliberately does NOT touch torch/numpy -- Colab
# ships a CUDA-matched torch preinstalled, and a naive `pip install torch`
# risks silently replacing it with a mismatched/CPU wheel. Only the two deps
# this unit actually adds beyond the base environment get installed.
set -e
cd "$(dirname "$0")"
python3 -m pip install --quiet sentencepiece "datasets>=2.18"
exec python3 train.py
