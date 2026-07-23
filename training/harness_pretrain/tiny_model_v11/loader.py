"""Load TinyModel v11 from an artefact directory.

An artefact dir (e.g. `tiny-model/model/v11/`) contains:
    config.json                # arch + metadata
    artifacts/model_compiled.pt # frozen-FFN attention-retrained weights
    artifacts/model_full.pt     # end of Phase 2 training
    artifacts/train_mask.pt     # optional — training-token mask
    vindex/                     # optional — extracted vindex

Most callers want `load_from_artifacts(path)`, which constructs the
model from `config.json` and loads the checkpoint of their choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .model import TinyModel


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    n_layers: int
    ffn_dim: int
    n_heads: int
    n_kv_heads: int
    max_seq: int
    version: str = "v11"
    notes: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["notes"] = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        out = {
            "version": self.version,
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "max_seq": self.max_seq,
        }
        out.update(self.notes)
        return out


def load_config(artefact_dir: str | Path) -> ModelConfig:
    path = Path(artefact_dir) / "config.json"
    with open(path) as f:
        return ModelConfig.from_dict(json.load(f))


def load_from_artifacts(
    artefact_dir: str | Path,
    checkpoint: str = "model_compiled.pt",
    device: Optional[str | torch.device] = None,
    strict: bool = True,
) -> tuple[TinyModel, ModelConfig]:
    """Instantiate TinyModel from a v11-style artefact dir.

    Returns (model, config). The model is set to eval() and moved to
    `device` (auto-selects MPS → CUDA → CPU if None).
    """
    root = Path(artefact_dir)
    config = load_config(root)

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = TinyModel(
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        max_seq=config.max_seq,
    ).to(device)

    ckpt_path = root / "artifacts" / checkpoint
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, config
