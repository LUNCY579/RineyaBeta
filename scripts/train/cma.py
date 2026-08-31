"""Gauge-correct low-rank Delta-CMA algebra for RGE-PEFT."""

from __future__ import annotations

import math

import torch

CMA_IMPLEMENTATION_VERSION = "rge-delta-cma-v1"


@torch.no_grad()
def concatenate_weighted_deltas(
    factors: list[tuple[torch.Tensor, torch.Tensor, float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Bc, Ac`` with ``Bc @ Ac = sum_i w_i s_i B_i @ A_i``."""
    if not factors:
        raise ValueError("Delta-CMA requires at least one checkpoint")
    bs, aas = [], []
    shape_b, shape_a = factors[0][0].shape, factors[0][1].shape
    for b, a, scale, weight in factors:
        if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError("CMA factors must be compatible matrices")
        if b.shape[0] != shape_b[0] or a.shape[1] != shape_a[1]:
            raise ValueError("CMA checkpoint target shapes differ")
        coefficient = float(scale) * float(weight)
        if coefficient < 0.0 or not math.isfinite(coefficient):
            raise ValueError("CMA v1 requires finite non-negative weight*scale")
        root = math.sqrt(coefficient)
        bs.append(b.detach().float() * root)
        aas.append(a.detach().float() * root)
    return torch.cat(bs, dim=1), torch.cat(aas, dim=0)


@torch.no_grad()
def qr_core_svd(bc: torch.Tensor, ac: torch.Tensor):
    """SVD of a low-rank product through thin QR and a small core."""
    if bc.ndim != 2 or ac.ndim != 2 or bc.shape[1] != ac.shape[0]:
        raise ValueError("Bc/Ac are incompatible")
    qb, rb = torch.linalg.qr(bc.float(), mode="reduced")
    qa, ra = torch.linalg.qr(ac.float().mT, mode="reduced")
    core = rb @ ra.mT
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    return qb, qa, u, singular, vh


@torch.no_grad()
def rank_compress(
    bc: torch.Tensor, ac: torch.Tensor, *, rank: int, output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return PEFT ``B,A`` whose scaled product is the rank-r CMA delta."""
    if rank < 1 or not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("rank/output_scale must be positive")
    qb, qa, u, singular, vh = qr_core_svd(bc, ac)
    kept = min(rank, singular.numel())
    root = singular[:kept].clamp_min(0.0).sqrt()
    split_scale = math.sqrt(float(output_scale))
    b = (qb @ u[:, :kept]) * root.unsqueeze(0) / split_scale
    a = root.unsqueeze(1) * (vh[:kept] @ qa.mT) / split_scale
    if kept < rank:
        b = torch.nn.functional.pad(b, (0, rank - kept))
        a = torch.nn.functional.pad(a, (0, 0, 0, rank - kept))
    total_energy = float(singular.square().sum().item())
    retained_energy = float(singular[:kept].square().sum().item())
    diagnostics = {
        "combined_rank_upper_bound": int(bc.shape[1]),
        "small_core_rank": int(singular.numel()),
        "retained_rank": int(rank),
        "total_frobenius_energy": total_energy,
        "retained_frobenius_energy": retained_energy,
        "retained_energy_fraction": (retained_energy / total_energy
                                     if total_energy else 1.0),
        "relative_reconstruction_error": math.sqrt(
            max(total_energy - retained_energy, 0.0) / total_energy)
            if total_energy else 0.0,
        "singular_values": singular.tolist(),
    }
    return b.contiguous(), a.contiguous(), diagnostics
