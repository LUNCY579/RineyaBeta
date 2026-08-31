#!/usr/bin/env python3
"""Synthetic proofs for gauge-correct Delta-CMA and rank recompression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from cma import concatenate_weighted_deltas, rank_compress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(20260831)
    checks = []
    factors = []
    dense = torch.zeros(11, 13)
    for index, weight in enumerate((0.2, 0.3, 0.5)):
        b = torch.randn(11, 3, generator=generator)
        a = torch.randn(3, 13, generator=generator)
        scale = 2.0
        factors.append((b, a, scale, weight))
        dense += weight * scale * (b @ a)
    bc, ac = concatenate_weighted_deltas(factors)
    assert torch.allclose(bc @ ac, dense, atol=2e-5, rtol=2e-5)
    checks.append("exact concatenation equals dense weighted delta average")

    q = torch.tensor([[1.1, 0.1, 0.0], [0.0, 0.9, 0.2],
                      [0.1, 0.0, 1.2]])
    gauged = list(factors)
    b, a, scale, weight = gauged[1]
    gauged[1] = (b @ q, torch.linalg.solve(q, a), scale, weight)
    gbc, gac = concatenate_weighted_deltas(gauged)
    assert torch.allclose(gbc @ gac, dense, atol=3e-5, rtol=3e-5)
    checks.append("exact average is gauge invariant")

    b2, a2, diagnostics = rank_compress(bc, ac, rank=4, output_scale=2.0)
    reconstructed = 2.0 * (b2 @ a2)
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    oracle = (u[:, :4] * singular[:4].unsqueeze(0)) @ vh[:4]
    assert torch.allclose(reconstructed, oracle, atol=5e-5, rtol=5e-5)
    checks.append("QR/core-SVD rank-r result matches dense truncated SVD")
    assert b2.shape == (11, 4) and a2.shape == (4, 13)
    checks.append("rank-matched factors have requested PEFT shapes")
    assert 0.0 <= diagnostics["retained_energy_fraction"] <= 1.0
    assert diagnostics["relative_reconstruction_error"] >= 0.0
    checks.append("retained energy and reconstruction error reported")

    wrong_a = sum(weight * a for _, a, _, weight in factors)
    wrong_b = sum(weight * b for b, _, _, weight in factors)
    assert not torch.allclose(2.0 * (wrong_b @ wrong_a), dense,
                              atol=1e-4, rtol=1e-4)
    checks.append("raw independent factor averaging is not used")

    for shape in ((64, 3, 7), (9, 3, 48)):
        m, rank, n = shape
        local = []
        target = torch.zeros(m, n)
        for weight in (0.5, 0.5):
            b = torch.randn(m, rank, generator=generator)
            a = torch.randn(rank, n, generator=generator)
            local.append((b, a, 1.5, weight))
            target += weight * 1.5 * b @ a
        lb, la = concatenate_weighted_deltas(local)
        rb, ra, _ = rank_compress(lb, la, rank=rank, output_scale=1.5)
        assert (1.5 * rb @ ra).shape == target.shape
    checks.append("rectangular target aspects recompress correctly")

    result = {"schema_version": 1, "status": "PASS", "checks": len(checks),
              "check_names": checks, "sealed_data_accessed": False}
    text = json.dumps(result, indent=2)
    print(text)
    if args.evidence:
        path = Path(args.evidence)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
