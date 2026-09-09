"""Position-explicit attention and parameter-matched, bounded query feedback."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("baseline", "adapter", "independent", "feedback")


@dataclass(frozen=True)
class Spec:
    arm: str = "feedback"
    layer: int = 14
    rank: int = 8
    backend: str = "eager"

    def __post_init__(self):
        if self.arm not in ARMS or self.backend not in ("eager", "sdpa"):
            raise ValueError("Unknown arm/backend")
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")


def rotate(x, cos, sin):
    """Qwen3 full-dimensional RoPE, applied exactly once to each fresh query."""
    if x.shape[-1] % 2 or cos.shape != sin.shape or cos.shape[-1] != x.shape[-1]:
        raise ValueError("Invalid full RoPE dimensions")
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


def attend(q, k, v, qp, kp, *, backend="eager", scale=None):
    """[B,H,T,D]; positions are explicit, including non-square cached reads.

    True means allowed. No dropout, padding, local windows, or implicit causal
    alignment. GQA repeats only in this reference implementation (not a speed claim).
    """
    if backend not in ("eager", "sdpa"):
        raise ValueError("Unknown attention backend")
    if any(t.ndim != 4 for t in (q, k, v)) or k.shape != v.shape:
        raise ValueError("Expected rank-four tensors with equal K/V shapes")
    if (q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]
            or min(q.shape + k.shape) < 1 or q.shape[1] % k.shape[1]):
        raise ValueError("Invalid batch, head, sequence, or GQA dimensions")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("Q/K/V must have the same device and dtype")
    if not q.is_floating_point():
        raise ValueError("Floating point attention required")
    for p, n in ((qp, q.shape[-2]), (kp, k.shape[-2])):
        if (p.shape != (n,) or p.dtype != torch.long or p.device != q.device
                or bool((p < 0).any()) or bool((p[1:] <= p[:-1]).any())):
            raise ValueError("Positions must be nonnegative, strictly increasing int64")
    allowed = kp[None, :] <= qp[:, None]
    if not bool(allowed.any(-1).all()):
        raise ValueError("Every query must have a visible key")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid attention scale")
    repeat = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                             dropout_p=0.0, is_causal=False, scale=scale)
    logits = (q @ k.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~allowed, -torch.inf)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    weights = torch.softmax(logits, dim=-1, dtype=dtype).to(q.dtype)
    return weights @ v


class Update(nn.Module):
    """Head-shared low-rank map plus per-head bounded gain: 2*D*r + H params.

    Zero UP alone ensures identity. DOWN and gain start nonzero, so UP receives
    gradients immediately; the other parameters can learn after the first update.
    This is a bounded finite-depth update, not a convergence guarantee.
    """
    def __init__(self, dim, heads, rank, *, device=None, dtype=None):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, dim, bias=False, device=device, dtype=dtype)
        self.gain = nn.Parameter(torch.full((heads,), 0.1, device=device, dtype=dtype))
        nn.init.zeros_(self.up.weight)

    def forward(self, source):
        dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
        z = source.to(dtype)
        z = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(source.dtype)
        delta = torch.tanh(self.up(F.silu(self.down(z))))
        return delta * torch.tanh(self.gain)[None, :, None, None]


def refined_read(q_unrotated, k_rotated, v, qp, kp, cos, sin, spec, updater=None,
                 *, scale=None):
    """Memory is identical between reads. Query updates occur AFTER QK-Norm,
    BEFORE RoPE. Second read is live in both two-read controls and feedback arms.
    """
    def read(q):
        return attend(rotate(q, cos, sin), k_rotated, v, qp, kp,
                      backend=spec.backend, scale=scale)
    first = read(q_unrotated)
    if spec.arm == "baseline":
        return first
    if updater is None:
        raise ValueError("Nonbaseline arms require an updater")
    if spec.arm == "adapter":
        return first + updater(first)
    source = q_unrotated if spec.arm == "independent" else first
    second = read(q_unrotated + updater(source))
    # Identical blend in the matched controls; both reads have an output path.
    return (first + second) * 0.5
