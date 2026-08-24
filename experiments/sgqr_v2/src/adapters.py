#!/usr/bin/env python3
"""Arm I: a generic learned MLP at the retired layer's own site, at matched bytes.

Why this is the sharper capacity control
----------------------------------------
Same-byte LoRA (arm H) puts generic capacity *elsewhere in the surviving
network*. That answers "could the same bytes spent anywhere do the job?" -- a
real question, but it leaves a gap: LoRA differs from the remnant in **placement**
as well as in **origin**, so a win for the remnant could be about the insertion
point rather than about the weights being the deleted ones.

This arm closes that gap. It installs, at exactly the position the deleted MLP
occupied, a gated MLP of exactly the same architecture family, with exactly the
same byte budget, but with weights **learned from scratch** instead of preserved:

    F_learned(x) = W_down [ silu(W_gate x) * (W_up x) ],  d_ff chosen for bytes

So the only remaining difference between arm C and arm I is the **origin of the
weights**. If the frozen 4-bit copy of the original d_ff=3072 MLP beats a trained
bf16 d_ff=1206 MLP at the same storage cost, preservation is buying something
that capacity at the same site does not. If it does not, the honest reading is
that the site and the budget were doing the work.

The byte arithmetic
-------------------
One INT4 remnant is 4,939,872 B. A bf16 gated MLP with hidden `h` and bottleneck
`m` stores `3 * h * m` parameters at 2 B each, so

    m = 4,939,872 / (2 * 3 * 1024) = 803.9  ->  m = 803  (99.9% of budget)

The learned MLP therefore gets a bottleneck of 803 against the original's 3072 --
about 26% of the width, which is exactly what a 4-bit-versus-16-bit storage trade
buys. That asymmetry is the point of the comparison, not a handicap: both arms
cost the same to ship.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GenericMlpAdapter(nn.Module):
    """A trained-from-scratch gated MLP occupying a retired layer's MLP slot.

    Same algebra as the donor it replaces, bias-free for the same reason (a
    ``down_proj`` bias would break the additive decomposition that every other
    arm relies on, so leaving it out keeps the arms comparable).

    Initialised so the module starts as an exact no-op: ``down`` is zeroed, which
    makes the freshly installed adapter numerically identical to deletion. Any
    measured difference is therefore attributable to training rather than to
    initialisation -- the same discipline the CSGQR alignment lens used.
    """

    def __init__(self, hidden: int, bottleneck: int,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.gate_proj = nn.Linear(hidden, bottleneck, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, bottleneck, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(bottleneck, hidden, bias=False, dtype=dtype)
        std = hidden ** -0.5
        nn.init.normal_(self.gate_proj.weight, std=std)
        nn.init.normal_(self.up_proj.weight, std=std)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    def stored_bytes(self, dtype_bytes: int = 2) -> int:
        return 3 * self.hidden * self.bottleneck * dtype_bytes

    def extra_repr(self) -> str:
        return (str(self.hidden) + "->" + str(self.bottleneck) + "->"
                + str(self.hidden) + ", " + format(self.stored_bytes(), ",") + " B")


def bottleneck_for_budget(budget_bytes: int, hidden: int,
                          dtype_bytes: int = 2) -> int:
    """Largest bottleneck whose stored bytes do not exceed ``budget_bytes``.

    Rounded **down** so the control never gets more bytes than the remnant it is
    compared against. Giving the baseline a shortfall of at most one bottleneck
    unit is the conservative direction: it cannot be accused of an unfair
    advantage, and the residual gap is reported.
    """
    return max(1, budget_bytes // (3 * hidden * dtype_bytes))


def install_adapters(model, layers, hidden: int, bottleneck: int, *,
                     state: dict | None = None, device="cuda:0",
                     dtype: torch.dtype = torch.bfloat16) -> dict:
    """Replace each retired layer's MLP with a ``GenericMlpAdapter``.

    ``state`` loads previously trained weights; without it the adapters start as
    exact no-ops, i.e. identical to four-layer deletion.
    """
    mods: dict[int, GenericMlpAdapter] = {}
    for l in layers:
        a = GenericMlpAdapter(hidden, bottleneck, dtype=dtype)
        if state is not None:
            a.load_state_dict({k.split(str(l) + ".", 1)[-1]: v
                               for k, v in state.items()
                               if k.startswith("layer" + str(l) + ".")})
        mods[l] = a.to(device)
        model.model.layers[l].mlp = mods[l]
    return mods


def adapter_state(mods: dict) -> dict:
    """Flatten per-layer adapters into one prefixed state dict."""
    out = {}
    for l, a in mods.items():
        for k, v in a.state_dict().items():
            out["layer" + str(l) + "." + k] = v.detach().to(torch.bfloat16).cpu()
    return out
