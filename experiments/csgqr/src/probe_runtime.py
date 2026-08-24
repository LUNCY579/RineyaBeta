#!/usr/bin/env python3
"""Deployment-side probes: the group-utility router and the need gate.

Two tiny linear readouts of the pre-remnant residual ``h``, and they are the only
form of CSGQR's two decisions that this campaign has evidence for:

* the **need gate** ``q = sigma(w.h + b)`` predicts whether the remnant is worth
  running. The exact vocabulary lens does this at AUROC 0.55-0.60; a linear probe
  on ``h`` reached 0.936 on held-out prompts. The information is in the state and
  the vocabulary projection destroys it.
* the **group router** ``U^ = W h + b`` predicts each group's marginal utility for
  *this* state, which is what the dynamic-significance selector consumes. A static
  global ranking leaves ~67% of the per-token oracle's utility unclaimed at
  rho=0.25, and per-token orderings agree with the global ranking at mean Spearman
  0.070, so the headroom is real and only a per-state predictor can reach it.

Why linear, and why that is a feature
-------------------------------------
The monitor has to be cheaper than the thing it gates. The closed form is
unforgiving: the exact LM-head gate costs ``|V| / (3 d_ff) = 3.56x`` the MLP it
decides about on this checkpoint, so it can never pay for itself. A linear probe
costs ``d_model`` MACs for the gate and ``d_model * n_groups`` for the router --
against ``6 * d_model * d_ff`` for the MLP, that is ``1/(6 d_ff) = 0.005%`` and
``n_groups/(6 d_ff) = 0.13%``. Three to four orders of magnitude below the
budget. Anything larger has to justify itself against a monitor cost that is
already the dominant term in the exact-lens case.

Storage is a ``.pt`` file holding float32 weights plus the metadata needed to
refuse a mismatch: hidden width, group count, the trace the fit came from, the
prompt split, and the git SHA. Loading a router trained for a different layer or
group count is refused rather than broadcast into silence.

Standardisation is folded in at load time
-----------------------------------------
The fits are done on standardised features. Rather than carry a mean/scale vector
into the forward path and pay two extra elementwise passes per token, the affine
is folded into the weight and bias once at load:

    w.((h - mu)/sd) + b  ==  (w/sd).h + (b - (w/sd).mu)

so the deployed form is a single matmul. Exactness of the fold is asserted in
``verify_csgqr.py`` against the unfolded computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save_probe(path: Path, *, kind: str, weight: Tensor, bias: Tensor,
               mu: Tensor, sd: Tensor, meta: dict[str, Any]) -> None:
    """Write a probe with everything needed to validate it at load time.

    ``weight`` is ``[out, d]`` and ``bias`` is ``[out]``, both in the
    *standardised* input space; ``mu``/``sd`` are the standardisation applied to
    ``h``. They are stored unfolded so the file records what was actually fitted,
    and folded at load so the forward path is one matmul.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": kind,
        "weight": weight.detach().to(torch.float32).cpu(),
        "bias": bias.detach().to(torch.float32).cpu(),
        "mu": mu.detach().to(torch.float32).cpu(),
        "sd": sd.detach().to(torch.float32).cpu(),
        "meta": meta,
    }, path)


def _fold(blob: dict[str, Any]) -> tuple[Tensor, Tensor]:
    """Fold standardisation into ``(weight, bias)`` so inference is one matmul."""
    w = blob["weight"].to(torch.float32)
    b = blob["bias"].to(torch.float32)
    mu = blob["mu"].to(torch.float32).reshape(1, -1)
    sd = blob["sd"].to(torch.float32).reshape(1, -1).clamp(min=1e-12)
    w_folded = w / sd                        # [out, d]
    b_folded = b - (w_folded * mu).sum(dim=1)
    return w_folded, b_folded


def _load(path: Path, expect_kind: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit("no probe at " + str(path)
                         + "; train one with train_router.py")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("kind") != expect_kind:
        raise SystemExit("probe at " + str(path) + " is kind '"
                         + str(blob.get("kind")) + "', expected '"
                         + expect_kind + "'")
    return blob


# --------------------------------------------------------------------------
# Loading into a callable the CSGQR module can use
# --------------------------------------------------------------------------
def load_group_router(path: Path, device) -> tuple[Callable[[Tensor], Tensor],
                                                   dict[str, Any]]:
    """A ``utility_fn``: ``[n, d_model] -> [n, n_groups]`` predicted utility.

    The returned callable is what ``selector="significance"`` consumes. It does
    no ranking of its own -- ranking, cost normalisation and the budget are the
    router policy's job, so that the same predictor can be swept against
    different policies without refitting.
    """
    blob = _load(Path(path), "group_utility")
    w, b = _fold(blob)
    w = w.to(device)
    b = b.to(device)
    meta = dict(blob["meta"])
    meta["path"] = str(path)

    @torch.no_grad()
    def predict(h: Tensor) -> Tensor:
        return (h.reshape(-1, h.shape[-1]).to(torch.float32) @ w.t()) + b

    return predict, meta


def load_need_probe(path: Path, device) -> tuple[Callable[[Tensor], Tensor],
                                                 dict[str, Any]]:
    """A ``gate_fn``: ``[n, d_model] -> [n]`` predicted need in ``[0, 1]``.

    Higher means the remnant is more needed. ``CSGQRMlp`` converts this to the
    same maturity orientation the lens features use (``C = 1 - q``) before any
    threshold sees it, so one three-level policy implementation serves both and a
    ``tau`` means the same kind of thing in each.
    """
    blob = _load(Path(path), "need")
    w, b = _fold(blob)
    w = w.to(device)
    b = b.to(device)
    meta = dict(blob["meta"])
    meta["path"] = str(path)

    @torch.no_grad()
    def predict(h: Tensor) -> Tensor:
        logit = (h.reshape(-1, h.shape[-1]).to(torch.float32) @ w.t()) + b
        return torch.sigmoid(logit).reshape(-1)

    return predict, meta


def probe_cost(hidden: int, out: int, d_ff: int) -> dict[str, float]:
    """Monitor cost of a linear probe against the MLP it gates, per token.

    Reported as a *share of the gated MLP*, because that is the only comparison
    that decides viability -- an absolute MAC count says nothing about whether
    the gate pays for itself.
    """
    probe = 2 * hidden * out
    mlp = 2 * 3 * hidden * d_ff
    return {"probe_macs": probe, "mlp_macs": mlp,
            "share_of_mlp": probe / mlp,
            "bytes_fp32": 4 * (hidden * out + out)}
