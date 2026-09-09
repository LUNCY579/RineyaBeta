"""Explicit Qwen3 decoder traversal; no global patches or source-weight surgery.

The native full-attention forward remains the independent implementation oracle.
A successful structural check alone does NOT qualify a checkpoint/runtime pair.
"""
import torch
from torch import nn
from .attention import CompactKV, blocked_attention
from .config import Spec


# Transformers 4.x stored None for "no RoPE scaling". Transformers >=5 instead
# materializes an explicit descriptor, {"rope_type": "default", "rope_theta": T},
# and drops rope_theta from the config root. Both encode the SAME unscaled RoPE,
# so a bare "is not None" test rejects an unmodified checkpoint on the newer
# library. Accept only that descriptor; every real scaling family (linear,
# dynamic, yarn, llama3, longrope) and any extra scaling parameter still fails.
UNSCALED_ROPE_FIELDS = frozenset({"rope_type", "type", "rope_theta"})


def rope_is_unscaled(config):
    scaling = getattr(config, "rope_scaling", None)
    if scaling is None:
        return True
    if not isinstance(scaling, dict):
        return False
    kind = scaling.get("rope_type", scaling.get("type"))
    return kind == "default" and set(scaling) <= UNSCALED_ROPE_FIELDS


def rope_theta(config):
    """Effective RoPE base, wherever this Transformers version records it.

    Returns None when the two locations disagree, so a caller fails closed
    rather than reporting a base the model does not actually use.
    """
    scaling = getattr(config, "rope_scaling", None)
    nested = scaling.get("rope_theta") if isinstance(scaling, dict) else None
    root = getattr(config, "rope_theta", None)
    values = [v for v in (root, nested) if v is not None]
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return None
    return values[0] if len(set(values)) == 1 else None


class HeadGate(nn.Module):
    def __init__(self, hidden_size, heads, *, device=None, dtype=None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, heads, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.proj.weight)

    def forward(self, normalized_hidden, attention_output):
        factor = 2 * torch.sigmoid(self.proj(normalized_hidden).float())
        return attention_output * factor.to(attention_output.dtype).transpose(1, 2).unsqueeze(-1)


def validate_model(model, spec):
    c = model.config
    if getattr(c, "model_type", None) != "qwen3":
        raise ValueError("Only a Qwen3 backbone is supported in v1")
    if getattr(c, "quantization_config", None) or getattr(model, "is_quantized", False):
        raise ValueError("Quantized/PEFT/custom runtimes require separate qualification")
    if not rope_is_unscaled(c):
        raise ValueError("v1 requires original unscaled RoPE; rope_scaling declares a scaled variant")
    if rope_theta(c) is None:
        raise ValueError("Missing, non-numeric or internally inconsistent RoPE theta")
    if getattr(c, "use_sliding_window", False) or getattr(c, "attention_dropout", 0) != 0:
        raise ValueError("v1 requires full attention and zero dropout")
    if any(t != "full_attention" for t in (getattr(c, "layer_types", None) or [])):
        raise ValueError("Source checkpoint already has non-global layer types")
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError("Existing forward hooks could be bypassed; refusing this backbone")
    body = model.model
    if len(body.layers) != spec.layers or c.num_hidden_layers != spec.layers:
        raise ValueError("Layer count does not match the checkpoint")
    h, kv, d, width = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
    if h % kv or d % 2:
        raise ValueError("Invalid GQA or rotary dimensions")
    for layer in body.layers:
        a = layer.self_attn
        for name, out in (("q_proj", h*d), ("k_proj", kv*d), ("v_proj", kv*d)):
            m = getattr(a, name)
            if not isinstance(m, nn.Linear) or tuple(m.weight.shape) != (out, width):
                raise ValueError(f"Unsupported {name} shape/module")
        if tuple(a.o_proj.weight.shape) != (width, h*d):
            raise ValueError("Unsupported output projection")
        for name in ("q_norm", "k_norm"):
            if tuple(getattr(a, name).weight.shape) != (d,):
                raise ValueError("QK-Norm must remain intact")
        for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
            if not hasattr(layer, name):
                raise ValueError(f"Missing Qwen3 component: {name}")
    if not hasattr(body, "rotary_emb") or not hasattr(body, "norm"):
        raise ValueError("Missing original rotary embedding or output norm")


def model_fingerprint(model):
    """Bind the LOADED module into a receipt, not merely the on-disk config.

    File hashes prove which bytes were read; this proves what the runtime
    actually instantiated from them under this Transformers version.
    """
    c, body = model.config, model.model
    return {
        "class": type(model).__name__,
        "model_type": getattr(c, "model_type", None),
        "layers": len(body.layers),
        "hidden_size": c.hidden_size,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "head_dim": c.head_dim,
        "intermediate_size": c.intermediate_size,
        "vocab_size": c.vocab_size,
        "max_position_embeddings": c.max_position_embeddings,
        "rope_theta_effective": rope_theta(c),
        "rope_unscaled": rope_is_unscaled(c),
        "attn_implementation": getattr(c, "_attn_implementation", None),
        "attention_scaling": sorted({float(l.self_attn.scaling) for l in body.layers}),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "distinct_parameter_count": sum(p.numel() for p in model.parameters()),
        "tied_output_embedding": bool(
            model.lm_head.weight.data_ptr() == body.embed_tokens.weight.data_ptr()),
    }


class HybridRuntime(nn.Module):
    def __init__(self, model, spec: Spec):
        super().__init__()
        validate_model(model, spec)
        self.backbone, self.spec = model, spec
        c = model.config
        w = model.model.embed_tokens.weight
        self.gates = nn.ModuleList([
            HeadGate(c.hidden_size, c.num_attention_heads, device=w.device, dtype=w.dtype)
            for _ in range(spec.layers)
        ] if spec.gate else [])
        # Tracing survives reset(): the two schedules under comparison each call
        # reset(), and the trace is what distinguishes them.
        self.trace = None
        self.reset()

    def start_trace(self):
        """Record the exact key positions each query attends to, per layer."""
        self.trace = [[] for _ in range(self.spec.layers)]
        return self.trace

    def stop_trace(self):
        self.trace = None

    def reset(self):
        self.cache = [CompactKV(self.spec.local_window(i),
                               self.spec.sinks if self.spec.local_window(i) is not None else 0)
                      for i in range(self.spec.layers)]
        self.position = 0
        self.stats = {}
        self.aborted = False

    def _abort(self):
        """Clear the cache AND refuse further cached use until reset() is called.

        reset() alone puts self.position back to 0, so a caller that catches a
        streaming error and retries the same chunk would silently restart
        absolute positions at zero and get a plausible but wrong continuation.
        """
        self.reset()
        self.aborted = True

    def forward(self, input_ids, *, use_cache=False, last_only=False):
        # Validation, embeddings and RoPE can fail before the decoder loop too.
        # Never leave a partially consumed request available for a later append.
        try:
            return self._forward_impl(input_ids, use_cache=use_cache, last_only=last_only)
        except Exception:
            if use_cache:
                self._abort()
            raise

    def _forward_impl(self, input_ids, *, use_cache=False, last_only=False):
        if use_cache and self.aborted:
            raise RuntimeError("Cached stream was aborted; call reset() before reusing this runtime")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("v1 accepts one nonempty, unpadded sequence; batch/packing are unsupported")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64")
        if use_cache and (torch.is_grad_enabled() or self.backbone.training):
            raise RuntimeError("Streaming requires eval mode and disabled autograd")
        c = self.backbone.config
        n = input_ids.shape[1]
        start = self.position if use_cache else 0
        if start + n > c.max_position_embeddings:
            raise ValueError("No context extension is authorized by this runtime")
        pos = torch.arange(start, start+n, device=input_ids.device)
        body = self.backbone.model
        x = body.embed_tokens(input_ids)
        cos, sin = body.rotary_emb(x, pos.unsqueeze(0))
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        if cos.shape[-1] != c.head_dim or sin.shape != cos.shape:
            raise ValueError("Partial/dual RoPE is deferred; source must use full rotary dimensions")
        h, kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim

        def rotate(t):
            a, b = t.chunk(2, dim=-1)
            return t*cos + torch.cat((-b, a), dim=-1)*sin

        # On any failed streaming forward, clear partially updated layer caches.
        try:
            for i, layer in enumerate(body.layers):
                z = layer.input_layernorm(x)
                a = layer.self_attn
                q = a.q_norm(a.q_proj(z).view(1, n, h, d)).transpose(1, 2)
                k = a.k_norm(a.k_proj(z).view(1, n, kv, d)).transpose(1, 2)
                v = a.v_proj(z).view(1, n, kv, d).transpose(1, 2)
                q, k = rotate(q), rotate(k)
                keys, values, kp = self.cache[i].merge(k, v, pos) if use_cache else (k, v, pos)
                window = self.spec.local_window(i)
                out = blocked_attention(
                    q, keys, values, pos, kp, window=window,
                    sinks=self.spec.sinks if window is not None else 0,
                    block_size=self.spec.block_size, backend=self.spec.backend,
                    scale=getattr(a, "scaling", d**-0.5), stats=self.stats,
                    trace=None if self.trace is None else self.trace[i],
                )
                if self.spec.gate:
                    out = self.gates[i](z, out)
                x = x + a.o_proj(out.transpose(1, 2).reshape(1, n, h*d))
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                if use_cache:
                    self.cache[i].retain(keys, values, kp)
            x = body.norm(x)
            logits = self.backbone.lm_head(x[:, -1:] if last_only else x)
            if use_cache:
                # Check every prefill chunk as well as decode, before argmax can
                # silently turn NaN/Inf into an apparently valid generated token.
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Nonfinite streaming logits; generation invalid")
                self.position += n
            return logits
        except Exception:
            if use_cache:
                self._abort()
            raise

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=24, eos_token_ids=(), chunk_size=128):
        # Reset before validation, including when a caller abandons cached forward.
        self.reset()
        generated, stopped, stop_token = [], False, None
        try:
            if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
                    or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                    or input_ids.dtype != torch.long):
                raise ValueError("Generation requires one nonempty unpadded int64 sequence")
            if (type(max_new_tokens) is not int or type(chunk_size) is not int
                    or max_new_tokens < 1 or chunk_size < 1):
                raise ValueError("Generation and chunk budgets must be positive integers")
            if not isinstance(eos_token_ids, (tuple, list)) or any(
                    type(t) is not int or not 0 <= t < self.backbone.config.vocab_size
                    for t in eos_token_ids):
                raise ValueError("EOS must contain in-vocabulary integer token IDs")
            if self.backbone.training:
                raise RuntimeError("Call backbone.eval() first")
            if input_ids.shape[-1] + max_new_tokens > self.backbone.config.max_position_embeddings:
                raise ValueError("Prompt plus generation budget exceeds original context limit")
            for start in range(0, input_ids.shape[-1], chunk_size):
                logits = self(input_ids[:, start:start+chunk_size], use_cache=True, last_only=True)
            for step in range(max_new_tokens):
                token = int(logits[0, -1].argmax())
                generated.append(token)
                if token in eos_token_ids:
                    stopped, stop_token = True, token
                    break
                if step + 1 < max_new_tokens:
                    next_id = torch.tensor([[token]], device=input_ids.device, dtype=torch.long)
                    logits = self(next_id, use_cache=True, last_only=True)
            return {
                "token_ids": generated, "stopped_eos": stopped,
                "stop_token_id": stop_token,
                "budget_exhausted": not stopped and len(generated) == max_new_tokens,
                "cache": [x.metrics() for x in self.cache], "attention_stats": dict(self.stats),
            }
        finally:
            # Return only metrics, never live tensors retaining full allocations.
            self.reset()
