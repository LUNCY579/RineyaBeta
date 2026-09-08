"""Random tiny Qwen3-SHAPED test fixture, not Rineya and not HF integration evidence."""
from types import SimpleNamespace
import torch
from torch import nn


class RMS(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)).to(x.dtype)*self.weight


class Rotary(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("freq", 1 / (10000 ** (torch.arange(0, d, 2).float()/d)))

    def forward(self, x, positions):
        t = positions[..., None]*self.freq
        t = torch.cat((t, t), -1)
        return t.cos().to(x.dtype), t.sin().to(x.dtype)


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, 2*width, bias=False)
        self.up = nn.Linear(width, 2*width, bias=False)
        self.down = nn.Linear(2*width, width, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x))*self.up(x))


class TinyQwen(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            hidden_size=32, vocab_size=97, max_position_embeddings=2048,
            rope_scaling=None, use_sliding_window=False, attention_dropout=0.0)
        body = nn.Module()
        body.embed_tokens = nn.Embedding(97, 32)
        body.rotary_emb = Rotary(8)
        body.norm = RMS(32)
        body.layers = nn.ModuleList()
        for i in range(layers):
            layer = nn.Module()
            a = nn.Module()
            for name, n in (("q_proj", 32), ("k_proj", 16), ("v_proj", 16), ("o_proj", 32)):
                setattr(a, name, nn.Linear(32, n, bias=False))
            a.q_norm, a.k_norm, a.scaling = RMS(8), RMS(8), 8**-0.5
            layer.self_attn = a
            layer.input_layernorm = RMS(32)
            layer.post_attention_layernorm = RMS(32)
            layer.mlp = MLP(32)
            body.layers.append(layer)
        self.model = body
        self.lm_head = nn.Linear(32, 97, bias=False)
        self.eval()

    def forward(self, input_ids, use_cache=False):
        """Independent dense implementation; never calls the research attention helper."""
        if use_cache:
            raise ValueError("Fixture native oracle is deliberately uncached")
        n = input_ids.shape[1]
        x = self.model.embed_tokens(input_ids)
        cs, sn = self.model.rotary_emb(x, torch.arange(n).unsqueeze(0))
        def rope(t):
            a, b = t.chunk(2, -1)
            return t*cs[:, None] + torch.cat((-b, a), -1)*sn[:, None]
        for layer in self.model.layers:
            z, a = layer.input_layernorm(x), layer.self_attn
            q = rope(a.q_norm(a.q_proj(z).reshape(1, n, 4, 8)).transpose(1, 2))
            k = rope(a.k_norm(a.k_proj(z).reshape(1, n, 2, 8)).transpose(1, 2))
            v = a.v_proj(z).reshape(1, n, 2, 8).transpose(1, 2)
            k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
            mask = torch.ones(n, n, dtype=torch.bool).tril()
            scores = (q @ k.transpose(-2, -1))*a.scaling
            out = torch.softmax(scores.masked_fill(~mask, -torch.inf), -1) @ v
            x = x + a.o_proj(out.transpose(1, 2).reshape(1, n, 32))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))


class CharacterTokenizer:
    def encode(self, s, add_special_tokens=False):
        return list(s.encode("ascii"))
