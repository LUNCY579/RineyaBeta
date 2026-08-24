#!/usr/bin/env python3
"""Intermediate-confidence estimation at the pre-remnant boundary.

Where the signal comes from
---------------------------
A causal LM has no probability distribution at an intermediate depth until one
is constructed. Following the design, the pre-remnant residual ``h`` is pushed
through the model's own final norm and LM head::

    z~ = W_LM . N_f( A(h) )

with ``A`` an optional alignment lens. ``A = identity`` is the raw logit lens and
is the honest baseline for "how much signal is already there without training
anything"; a trained ``A`` is a separate, later arm, so that any confidence
result can be attributed to one or the other.

Capture point. In ``Qwen3DecoderLayer`` the MLP sub-block is::

    residual = hidden_states                       # post-attention residual
    hidden_states = post_attention_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)
    hidden_states = residual + hidden_states

so the state *immediately before* the MLP remnant is ``residual`` -- the
pre-norm, post-attention residual. That is what the lens reads. Reading the
post-norm tensor instead would be a different (and RMSNorm-rescaled) quantity;
``surgery.py`` takes ``residual`` via a forward-pre-hook on
``post_attention_layernorm``, whose input is exactly it.

Calibration temperature is not generation temperature
-----------------------------------------------------
``T`` here is a calibration parameter fitted on a held-out split to make a
stated confidence correspond to observed reliability. The generation protocol in
this repository is greedy (``do_sample=False``, ``temperature=null``), so no
sampling transform touches these logits -- but the two must not be conflated
even so, and this module never reads a generation config. Confidence is computed
from raw lens logits before any top-p/top-k/repetition transform, so changing a
user's decoding settings cannot change what the gate means.

Vocabulary tensors are not retained
-----------------------------------
Full-vocabulary projection is the known efficiency trap for confidence-gated
inference. :func:`features_from_hidden` projects, reduces to a handful of
scalars, and lets the logits die inside the call; it never returns them and
never stores them. ``[B, T, |V|]`` at ``|V| = 32768`` is 64 MB per 1000
positions in bf16, so this is the difference between a viable gate and an
unaffordable one -- ``bench_gpu.py`` measures what the projection still costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Feature names, in the order the analysis code expects them.
FEATURE_NAMES = ("max_prob", "neg_entropy", "prob_margin", "logit_margin")


class AlignmentLens(nn.Module):
    """``A`` in ``W_LM . N_f(A(h))``.

    ``rank=0`` is the identity -- a raw logit lens with no trained parameters,
    which is the reference arm. ``rank>0`` adds a low-rank residual
    ``h + U V h`` initialised to exactly zero, so a freshly constructed lens is
    numerically identical to the identity and any measured difference is
    attributable to training rather than to initialisation.
    """

    def __init__(self, hidden: int, rank: int = 0, *,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.hidden = hidden
        self.rank = rank
        if rank > 0:
            self.down = nn.Linear(hidden, rank, bias=False, dtype=dtype)
            self.up = nn.Linear(rank, hidden, bias=False, dtype=dtype)
            nn.init.normal_(self.down.weight, std=hidden ** -0.5)
            nn.init.zeros_(self.up.weight)

    def forward(self, h: Tensor) -> Tensor:
        if self.rank == 0:
            return h
        return h + self.up(self.down(h.to(self.up.weight.dtype))).to(h.dtype)

    def extra_repr(self) -> str:
        return "hidden=" + str(self.hidden) + ", rank=" + str(self.rank) + (
            " (identity / raw logit lens)" if self.rank == 0 else "")


@dataclass
class ConfidenceHead:
    """The pieces needed to turn a residual state into next-token logits.

    Holds borrowed references to the *model's own* final norm and LM head. It
    does not copy them, so this adds no resident weight bytes -- which matters,
    because a duplicated 32768x1024 head would be 64 MB in bf16 and would swamp
    the entire memory story of a single-layer remnant.
    """

    final_norm: nn.Module
    lm_head: nn.Module
    lens: AlignmentLens
    vocab_size: int
    temperature: float = 1.0

    def logits(self, h: Tensor) -> Tensor:
        """Raw lens logits for ``h``, before temperature.

        Exposed for calibration fitting and for the analysis passes. The
        deployment path must use :func:`features_from_hidden` instead, which
        never lets a vocabulary-sized tensor escape.
        """
        return self.lm_head(self.final_norm(self.lens(h)))


def features_from_logits(logits: Tensor, vocab_size: int | None = None,
                         temperature: float = 1.0) -> dict[str, Tensor]:
    """Reduce next-token logits to scalar confidence features.

    Computed in float32 regardless of the incoming dtype: a bf16 softmax over
    32768 classes loses enough precision in the tail that the entropy term moves
    in the third decimal, which is the same order as the differences between
    confidence bins that this study needs to resolve.

    Returns, per position:

    ``max_prob``      max_v p(v).
    ``neg_entropy``   1 - H(p)/log|V|, so it rises with confidence like the
                      others and lands in [0, 1]. The normaliser is ``log|V|``
                      over the *full* vocabulary, which is the maximum entropy
                      attainable; normalising by anything else (the observed
                      support, say) would make the value incomparable across
                      positions.
    ``prob_margin``   p(1) - p(2).
    ``logit_margin``  z(1) - z(2), unnormalised, and so not confined to [0, 1].
    ``top1``/``top2`` the argmax and runner-up ids, kept for the top-token
                      agreement analysis.
    """
    z = logits.to(torch.float32)
    if temperature != 1.0:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0; got " + str(temperature))
        z = z / temperature
    v = int(z.shape[-1]) if vocab_size is None else int(vocab_size)

    logp = torch.log_softmax(z, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    top2 = torch.topk(z, k=2, dim=-1)
    p2 = torch.topk(p, k=2, dim=-1).values

    return {
        "max_prob": p2[..., 0],
        "neg_entropy": 1.0 - entropy / math.log(v),
        "prob_margin": p2[..., 0] - p2[..., 1],
        "logit_margin": top2.values[..., 0] - top2.values[..., 1],
        "top1": top2.indices[..., 0],
        "top2": top2.indices[..., 1],
    }


@torch.no_grad()
def features_from_hidden(head: ConfidenceHead, h: Tensor) -> dict[str, Tensor]:
    """Confidence features for residual states ``h``, without retaining logits.

    The vocabulary-sized tensor exists only inside this function and is dropped
    on return. Callers get a handful of scalars per position -- tens of bytes,
    not ``|V|`` floats.
    """
    return features_from_logits(head.logits(h), head.vocab_size, head.temperature)


def js_divergence(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    """Jensen-Shannon divergence in nats between two logit tensors.

    Used for the prediction-stability feature ``C^S = 1 - JS(p_{j-d}, p_j)``:
    a prediction that is concentrated *and* stable across nearby depths is more
    plausibly mature than one that concentrates abruptly at a single layer. JS
    rather than KL because it is symmetric and bounded by log 2, so subtracting
    it from 1 gives a bounded feature.
    """
    lp = torch.log_softmax(logits_a.to(torch.float32), dim=-1)
    lq = torch.log_softmax(logits_b.to(torch.float32), dim=-1)
    p, q = lp.exp(), lq.exp()
    m = torch.log(0.5 * (p + q) + 1e-30)
    return 0.5 * ((p * (lp - m)).sum(-1) + (q * (lq - m)).sum(-1))


def stability(logits_shallow: Tensor, logits_deep: Tensor) -> Tensor:
    """``1 - JS/log2``, in [0, 1], rising with agreement across depths."""
    return 1.0 - js_divergence(logits_shallow, logits_deep) / math.log(2.0)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def fit_temperature(logits: Tensor, targets: Tensor, *,
                    max_iter: int = 200, lo: float = 0.05,
                    hi: float = 20.0) -> float:
    """Temperature minimising NLL of ``targets`` under ``softmax(logits/T)``.

    Golden-section search on a 1-D convex problem, which is more robust than
    gradient descent with a hand-tuned learning rate and needs no optimiser
    state. ``logits`` and ``targets`` must come from the *calibration* split
    only; :func:`expected_calibration_error` is then evaluated on a disjoint
    split. Nothing in this module looks at the sealed set.
    """
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits/targets length mismatch: "
                         + str(logits.shape[0]) + " vs " + str(targets.shape[0]))
    z = logits.to(torch.float32)
    y = targets.to(torch.long)

    def nll(t: float) -> float:
        return torch.nn.functional.cross_entropy(z / t, y).item()

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(max_iter):
        if b - a < 1e-6:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
    return 0.5 * (a + b)


def expected_calibration_error(confidence: Tensor, correct: Tensor,
                               n_bins: int = 15) -> dict[str, float]:
    """ECE, MCE and Brier score with equal-width bins.

    ``correct`` is a 0/1 tensor. Empty bins are skipped rather than counted as
    perfectly calibrated, which would flatter a confidence signal that never
    visits part of the range.
    """
    c = confidence.to(torch.float32).flatten()
    y = correct.to(torch.float32).flatten()
    if c.numel() != y.numel():
        raise ValueError("confidence/correct length mismatch")
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, n = 0.0, 0.0, c.numel()
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        sel = (c > lo) & (c <= hi) if i else (c >= lo) & (c <= hi)
        k = int(sel.sum().item())
        if not k:
            continue
        conf = c[sel].mean().item()
        acc = y[sel].mean().item()
        gap = abs(conf - acc)
        ece += (k / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": lo, "hi": hi, "n": k, "mean_confidence": conf,
                     "accuracy": acc, "gap": conf - acc})
    return {"ece": ece, "mce": mce,
            "brier": ((c - y) ** 2).mean().item(),
            "n": n, "bins": bins}
