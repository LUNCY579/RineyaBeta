"""Explicit, reversible Qwen3 traversal. The untouched HF forward is the oracle.

The traversal follows the repository's hybrid_attention_v1 integration contract,
not its attention implementation. No inheritance from its experiment or evidence.
"""
import torch
from torch import nn
from .core import Spec, Update, attend, refined_read, rotate


def validate_model(model, spec):
    c = model.config
    if getattr(c, "model_type", None) != "qwen3":
        raise ValueError("Only Qwen3 is supported")
    if getattr(model, "is_quantized", False) or getattr(c, "quantization_config", None):
        raise ValueError("Quantized/PEFT/custom backbones are not qualified")
    if (getattr(c, "rope_scaling", None) is not None
            or getattr(c, "use_sliding_window", False)
            or getattr(c, "attention_dropout", 0) != 0
            or any(t != "full_attention" for t in (getattr(c, "layer_types", None) or []))):
        raise ValueError("Requires original full attention, unscaled RoPE, zero dropout")
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError("Hooks could be bypassed by traversal")
    if hasattr(model, "peft_config") or getattr(model, "hf_device_map", None):
        raise ValueError("PEFT and dispatched models require separate qualification")
    h, kv, d, w = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
    if h % kv or d % 2 or len(model.model.layers) != c.num_hidden_layers:
        raise ValueError("Invalid native structure")
    if spec.layer >= c.num_hidden_layers:
        raise ValueError("Selected layer is out of range")
    for layer in model.model.layers:
        a = layer.self_attn
        for name, out in (("q_proj", h*d), ("k_proj", kv*d), ("v_proj", kv*d)):
            m = getattr(a, name)
            if type(m) is not nn.Linear or tuple(m.weight.shape) != (out, w):
                raise ValueError(f"Unsupported {name}")
        if type(a.o_proj) is not nn.Linear or tuple(a.o_proj.weight.shape) != (w, h*d):
            raise ValueError("Unsupported output projection")
        if any(tuple(getattr(a, name).weight.shape) != (d,) for name in ("q_norm", "k_norm")):
            raise ValueError("Original QK-Norm required")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("v1 qualification is float32 only; mixed precision is deferred")
    if len({p.device for p in model.parameters()}) != 1:
        raise ValueError("Single-device execution required")


class Runtime(nn.Module):
    def __init__(self, model, spec=Spec()):
        super().__init__()
        validate_model(model, spec)
        self.backbone, self.spec = model, spec
        c = model.config
        w = model.model.embed_tokens.weight
        self.update = None if spec.arm == "baseline" else Update(
            c.head_dim, c.num_attention_heads, spec.rank, device=w.device, dtype=w.dtype)
        self.reset()

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        return self

    def reset(self):
        self.cache = [None] * self.backbone.config.num_hidden_layers
        self.position = 0

    def forward(self, ids, *, use_cache=False, last_only=False):
        try:
            return self._forward(ids, use_cache=use_cache, last_only=last_only)
        except Exception:
            if use_cache:
                self.reset()
            raise

    def _forward(self, ids, *, use_cache, last_only):
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                or ids.shape[1] < 1 or ids.dtype != torch.long):
            raise ValueError("One nonempty unpadded int64 sequence is required")
        if use_cache and (torch.is_grad_enabled() or self.backbone.training):
            raise ValueError("Cache is inference-only: eval() and disabled autograd required")
        c, body = self.backbone.config, self.backbone.model
        n = ids.shape[1]
        start = self.position if use_cache else 0
        if start + n > c.max_position_embeddings:
            raise ValueError("Original context limit exceeded")
        pos = torch.arange(start, start+n, device=ids.device)
        x = body.embed_tokens(ids)
        cos, sin = body.rotary_emb(x, pos[None, :])
        cos, sin = cos[:, None], sin[:, None]
        h, kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        pending = []  # Transactional cache: commit only after every layer succeeds.
        for i, layer in enumerate(body.layers):
            z = layer.input_layernorm(x)
            a = layer.self_attn
            q = a.q_norm(a.q_proj(z).view(1, n, h, d)).transpose(1, 2)
            k = a.k_norm(a.k_proj(z).view(1, n, kv, d)).transpose(1, 2)
            v = a.v_proj(z).view(1, n, kv, d).transpose(1, 2)
            k = rotate(k, cos, sin)
            kp = pos
            if use_cache and self.cache[i] is not None:
                old_k, old_v, old_p = self.cache[i]
                k, v = torch.cat((old_k, k), -2), torch.cat((old_v, v), -2)
                kp = torch.cat((old_p, pos))
            scale = getattr(a, "scaling", d ** -0.5)
            if i == self.spec.layer:
                out = refined_read(q, k, v, pos, kp, cos, sin, self.spec, self.update, scale=scale)
            else:
                out = attend(rotate(q, cos, sin), k, v, pos, kp,
                             backend=self.spec.backend, scale=scale)
            x = x + a.o_proj(out.transpose(1, 2).reshape(1, n, h*d))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if use_cache:
                pending.append((k, v, kp))
        x = body.norm(x)
        logits = self.backbone.lm_head(x[:, -1:] if last_only else x)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite logits")
        if use_cache:
            self.cache, self.position = pending, start+n
        return logits

    @torch.inference_mode()
    def generate(self, ids, *, max_new_tokens=8, eos=(), chunk_size=128):
        self.reset()
        try:
            if type(max_new_tokens) is not int or max_new_tokens < 1:
                raise ValueError("Positive generation budget required")
            if type(chunk_size) is not int or chunk_size < 1:
                raise ValueError("Positive chunk size required")
            if any(type(t) is not int or not 0 <= t < self.backbone.config.vocab_size for t in eos):
                raise ValueError("Invalid EOS token")
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                    or ids.shape[1] < 1 or ids.dtype != torch.long
                    or ids.shape[1] + max_new_tokens > self.backbone.config.max_position_embeddings):
                raise ValueError("Invalid prompt or generation exceeds original context limit")
            for start in range(0, ids.shape[1], chunk_size):
                logits = self(ids[:, start:start+chunk_size], use_cache=True, last_only=True)
            result = []
            for step in range(max_new_tokens):
                token = int(logits[0, -1].argmax())
                result.append(token)
                if token in eos:
                    break
                if step+1 < max_new_tokens:
                    logits = self(torch.tensor([[token]], device=ids.device), use_cache=True, last_only=True)
            return result
        finally:
            self.reset()
