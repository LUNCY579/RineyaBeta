"""Bounded-query-block attention and compact inference KV storage.

Local blocks gather only their causal window union and persistent prefix.
Global layers remain quadratic in total FLOPs. This is a correctness-oriented
PyTorch implementation, NOT a native sparse/fused kernel or a speedup claim.
"""
import torch
from torch.nn import functional as F


def allowed_mask(query_positions, key_positions, window=None, sinks=0):
    causal = key_positions[None, :] <= query_positions[:, None]
    if window is None:
        return causal
    if not 0 <= sinks < window:
        raise ValueError("Invalid sink/window budget")
    recent = window - sinks
    keep = ((key_positions[None, :] < sinks)
            | (key_positions[None, :] >= query_positions[:, None] - recent + 1))
    return causal & keep


def _positions(p, size, device):
    if p.dtype != torch.long or p.ndim != 1 or len(p) != size or p.device != device:
        raise ValueError("Positions must be device-matched one-dimensional int64")
    if size == 0 or (p < 0).any() or (p[1:] <= p[:-1]).any():
        raise ValueError("Positions must be nonempty, nonnegative and strictly increasing")


def _validate(q, k, v, qp, kp):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected Q/K/V [batch, heads, tokens, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Batch/head dimension mismatch")
    if min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Q heads must be an integer multiple of KV heads")
    if not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must have a common floating dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V device mismatch")
    _positions(qp, q.shape[-2], q.device)
    _positions(kp, k.shape[-2], k.device)


def _attend(q, k, v, mask, backend, scale):
    if not mask.any(dim=-1).all():
        raise ValueError("All-masked query row; refusing silent NaNs")
    groups = q.shape[1] // k.shape[1]
    # Explicit expansion avoids the historical enable_gqa kernel confound.
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    if backend == "sdpa":
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[None, None], dropout_p=0.0,
            is_causal=False, scale=scale,
        )
    if backend != "math":
        raise ValueError("Unknown attention backend")
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return probs @ v


def dense_oracle(q, k, v, qp, kp, *, window=None, sinks=0, scale=None):
    """Full dense mask, for SMALL correctness fixtures only."""
    _validate(q, k, v, qp, kp)
    return _attend(q, k, v, allowed_mask(qp, kp, window, sinks), "math",
                   q.shape[-1] ** -0.5 if scale is None else scale)


def blocked_attention(q, k, v, qp, kp, *, window=None, sinks=0,
                      block_size=128, backend="math", scale=None, stats=None):
    _validate(q, k, v, qp, kp)
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be positive")
    if window is not None and (type(window) is not int or not 0 <= sinks < window):
        raise ValueError("Invalid sink/window budget")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs = []
    for start in range(0, len(qp), block_size):
        pos = qp[start:start + block_size]
        candidate = kp <= pos[-1]
        if window is not None:
            candidate &= (kp < sinks) | (kp >= pos[0] - (window - sinks) + 1)
        idx = candidate.nonzero(as_tuple=True)[0]
        p = kp.index_select(0, idx)
        mask = allowed_mask(pos, p, window, sinks)
        outputs.append(_attend(
            q[:, :, start:start + block_size], k.index_select(2, idx),
            v.index_select(2, idx), mask, backend, scale,
        ))
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["max_mask_elements"] = max(stats.get("max_mask_elements", 0), mask.numel())
            stats["max_key_tokens"] = max(stats.get("max_key_tokens", 0), len(p))
    return torch.cat(outputs, dim=2)


def storage_bytes(tensors):
    """Count de-duplicated resident storage, not just visible tensor elements."""
    stores = {}
    for t in tensors:
        if t is not None:
            s = t.untyped_storage()
            stores[(str(t.device), s.data_ptr())] = s.nbytes()
    return sum(stores.values())


class CompactKV:
    """Inference-only state. Eviction occurs AFTER all queries in a chunk attend."""
    def __init__(self, window=None, sinks=0):
        if (window is None and sinks != 0) or (
                window is not None and (type(window) is not int or not 0 <= sinks < window)):
            raise ValueError("Invalid cache policy")
        self.window, self.sinks = window, sinks
        self.k = self.v = self.positions = None
        self.seen = 0

    def merge(self, k, v, positions):
        if torch.is_grad_enabled():
            raise RuntimeError("Streaming cache is inference-only; use no_grad/inference_mode")
        if k.ndim != 4 or v.shape != k.shape or k.device != v.device or k.dtype != v.dtype:
            raise ValueError("Invalid K/V shape, dtype or device")
        _positions(positions, k.shape[2], k.device)
        expected = torch.arange(self.seen, self.seen + len(positions), device=k.device)
        if not torch.equal(positions, expected):
            raise ValueError("Non-contiguous append; reset between requests")
        if self.k is None:
            return k, v, positions
        if (k.shape[:2] != self.k.shape[:2] or k.shape[-1] != self.k.shape[-1]
                or k.dtype != self.k.dtype or k.device != self.k.device):
            raise ValueError("Cache shape/dtype/device changed")
        return (torch.cat((self.k, k), 2), torch.cat((self.v, v), 2),
                torch.cat((self.positions, positions)))

    def retain(self, k, v, positions):
        if torch.is_grad_enabled():
            raise RuntimeError("Cache retention is inference-only")
        keep = torch.ones_like(positions, dtype=torch.bool)
        if self.window is not None:
            keep = ((positions < self.sinks)
                    | (positions >= positions[-1] - (self.window - self.sinks) + 1))
        idx = keep.nonzero(as_tuple=True)[0]
        # index_select allocates compact independent storage. A trailing slice
        # alone would retain the entire prefill backing allocation.
        self.k = k.detach().index_select(2, idx)
        self.v = v.detach().index_select(2, idx)
        self.positions = positions.index_select(0, idx)
        self.seen = int(positions[-1]) + 1

    def metrics(self):
        ts = (self.k, self.v)
        return {
            "seen_tokens": self.seen,
            "retained_tokens": 0 if self.positions is None else len(self.positions),
            "logical_kv_bytes": sum(t.numel() * t.element_size() for t in ts if t is not None),
            "resident_kv_bytes": storage_bytes(ts),
            "position_bytes": storage_bytes((self.positions,)),
        }
