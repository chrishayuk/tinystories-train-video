"""TinyModel v11 architecture.

Decoder-only transformer (Gemma-shaped): RMSNorm, RoPE, GQA, gated FFN,
tied embeddings. Weights and vindex live under sibling `v11/`; training
code under `v11-train/`.
"""

from .model import (
    TinyModel,
    TransformerBlock,
    Attention,
    GatedFFN,
    RMSNorm,
    precompute_rope,
    apply_rope,
)
from .loader import load_from_artifacts, load_config, ModelConfig

__all__ = [
    "TinyModel",
    "TransformerBlock",
    "Attention",
    "GatedFFN",
    "RMSNorm",
    "precompute_rope",
    "apply_rope",
    "load_from_artifacts",
    "load_config",
    "ModelConfig",
]

__version__ = "0.1.0"
