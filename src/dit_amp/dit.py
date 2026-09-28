# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
"""DiT backbone with a Hydra-friendly config."""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

@dataclass
class DiTConfig:
    N: int = 28
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype=next(self.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps=1e-5):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(dim, 1024), 6 * dim, bias=True))

    def forward(self, x, freqs_cis, adaln_input=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True))
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DDiT_Llama(nn.Module):
    def __init__(self, N, dim, n_layers, n_heads):
        super().__init__()
        self.N = N

        self.embedder = nn.Embedding(N, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.cond_embedder = nn.Linear(2, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.final_layer = FinalLayer(dim, 1, N)
        self.freqs_cis = self.precompute_freqs_cis(dim // n_heads, 4096)

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return cls(N=cfg.N, dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads)

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, x, t, cond=None):
        self.freqs_cis = self.freqs_cis.to(x.device)
        x_onehot = torch.nn.functional.one_hot(x, self.N).to(x.device, dtype=next(self.parameters()).dtype)
        x = self.embedder(x)
        adaln_input = self.t_embedder(t)

        if cond is not None:
            cond = cond.to(device=x.device, dtype=adaln_input.dtype)
            adaln_input = adaln_input + self.cond_embedder(cond)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        return x + x_onehot
