"""Small independent synthetic oracle, NOT a pretrained or native HF checkpoint."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F


class Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class Rotary(nn.Module):
    def forward(self, x, positions):
        angles = positions.float()[..., None] * (10000 ** (-torch.arange(0, 8, 2, device=x.device).float()/8))
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos(), angles.sin()


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=16, head_dim=8,
            num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=3,
            vocab_size=37, max_position_embeddings=256, attention_dropout=0,
            rope_scaling=None, use_sliding_window=False, layer_types=["full_attention"]*3)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(37, 16)
        self.model.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Module()
            layer.input_layernorm, layer.post_attention_layernorm = Norm(16), Norm(16)
            layer.mlp = nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 16))
            a = nn.Module()
            a.q_proj, a.k_proj, a.v_proj = nn.Linear(16, 32), nn.Linear(16, 16), nn.Linear(16, 16)
            a.o_proj, a.q_norm, a.k_norm = nn.Linear(32, 16), Norm(8), Norm(8)
            a.scaling = 8**-0.5
            layer.self_attn = a
            self.model.layers.append(layer)
        self.model.norm, self.model.rotary_emb = Norm(16), Rotary()
        self.lm_head = nn.Linear(16, 37, bias=False)

    def forward(self, input_ids, use_cache=False):
        if use_cache:
            raise ValueError("Fixture oracle has no cache")
        x = self.model.embed_tokens(input_ids)
        n = x.shape[1]
        cos, sin = self.model.rotary_emb(x, torch.arange(n, device=x.device)[None])
        def rope(t):
            shifted = torch.cat([-t[..., 4:], t[..., :4]], -1)
            return t*cos[:, None] + shifted*sin[:, None]
        for layer in self.model.layers:
            z, a = layer.input_layernorm(x), layer.self_attn
            q = rope(a.q_norm(a.q_proj(z).reshape(1, n, 4, 8)).permute(0, 2, 1, 3))
            k = rope(a.k_norm(a.k_proj(z).reshape(1, n, 2, 8)).permute(0, 2, 1, 3))
            v = a.v_proj(z).reshape(1, n, 2, 8).permute(0, 2, 1, 3)
            # Independent oracle uses SDPA, not the tested attention helper.
            y = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                               v.repeat_interleave(2, 1), is_causal=True)
            x = x + a.o_proj(y.permute(0, 2, 1, 3).reshape(1, n, 32))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))
