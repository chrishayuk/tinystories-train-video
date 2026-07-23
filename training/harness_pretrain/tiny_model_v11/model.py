"""TinyModel v11 core architecture.

Decoder-only Gemma-shaped transformer: RMSNorm, RoPE, GQA, gated FFN,
tied embeddings. Shared across v11 weight variants (v11, v11a, …); a
new core ships only when the architecture itself changes (v12-core).

Default v11 shape: dim=512, layers=20, heads=8, kv=4, ffn=2048,
vocab=71261, max_seq=256. Authoritative per-weights values live in
`v11/config.json`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (batch, seq, heads, head_dim)
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    out = torch.view_as_real(xc * freqs).flatten(-2)
    return out.type_as(x)


class GatedFFN(nn.Module):
    """Gated FFN: out = down(silu(gate(x)) * up(x))"""
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.gqa_ratio = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # GQA: repeat KV heads
        if self.gqa_ratio > 1:
            k = k.repeat_interleave(self.gqa_ratio, dim=2)
            v = v.repeat_interleave(self.gqa_ratio, dim=2)

        # (B, heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product with causal mask
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = GatedFFN(dim, ffn_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 71261,
        dim: int = 512,
        n_layers: int = 20,
        ffn_dim: int = 2048,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        max_seq: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, ffn_dim, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed.weight

        # RoPE frequencies (not a parameter)
        head_dim = dim // n_heads
        self.register_buffer("rope_freqs", precompute_rope(head_dim, max_seq))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids) * math.sqrt(self.dim)  # sqrt(dim) embed scaling (Gemma-style)

        for layer in self.layers:
            x = layer(x, self.rope_freqs)

        x = self.norm(x)
        return self.lm_head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = TinyModel()
    print(f"Parameters: {model.param_count():,}")
    print(f"Layers: {model.n_layers}")
    print(f"Hidden dim: {model.dim}")
    print(f"FFN dim: {model.ffn_dim}")

    print(f"Vocab: {model.vocab_size}")

    x = torch.randint(0, model.vocab_size, (1, 64))
    logits = model(x)
    print(f"Input: {x.shape} → Logits: {logits.shape}")
