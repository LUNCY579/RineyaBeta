#!/usr/bin/env python3
"""Fit the two gate probes from the calibration trace, and score them on val.

Two targets, deliberately kept apart, because the historical campaign conflated
them and that is one of the things this audit is testing:

``need``        the *actual* per-token benefit of running the remnant,
                U_t = KL(dense||deleted)_t - KL(dense||remnant)_t. This is what
                a gate should predict if its job is to decide where recovery
                compute is worth spending. SGQR-v2's preregistration asks for
                exactly this re-targeting.
``confidence``  the model's own maturity at that token, here ``max_prob`` from
                the logit lens. This is what CSGQR-v1's probe predicted. It is
                fitted here too so that "confidence-gated" and "need-gated" can
                be compared as two gates rather than as a gate and a baseline.

Ridge regression in closed form on standardized features, with the ridge
strength chosen on the VALIDATION split, never on test. A linear probe is used
because CSGQR-v1's systems measurement is what makes any gate viable at all: a
d_model -> 1 GEMM costs ~12% of the gated MLP in wall clock while the exact lens
costs ~1400%, so a heavier probe would spend the saving it exists to create.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import REPO                                                # noqa: E402
from stats import pearson, spearman                                  # noqa: E402

RES = REPO / "experiments/audit_sgqr_csgqr/results"


def read_trace(split: str, bits: str = "int4"):
    rows = [json.loads(l) for l in
            (RES / ("trace_%s_%s.jsonl" % (split, bits))).read_text(
                encoding="utf-8").splitlines() if l.strip()]
    hid = torch.load(RES / ("trace_%s_%s.hidden.pt" % (split, bits)),
                     weights_only=False)["hidden"]
    if hid.shape[0] != len(rows):
        raise SystemExit("hidden rows %d != trace rows %d" % (hid.shape[0], len(rows)))
    return rows, hid.to(torch.float64)


def ridge(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
    A = Xb.T @ Xb
    A += lam * torch.eye(d + 1, dtype=X.dtype)
    A[d, d] -= lam                      # do not penalise the intercept
    return torch.linalg.solve(A, Xb.T @ y)


def apply_w(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1) @ w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="int4")
    ap.add_argument("--lams", default="1,10,100,1000,10000,100000")
    ap.add_argument("--out", default="experiments/audit_sgqr_csgqr/results/probes.pt")
    args = ap.parse_args()

    tr_rows, tr_h = read_trace("calib", args.bits)
    va_rows, va_h = read_trace("val", args.bits)

    mu, sd = tr_h.mean(dim=0), tr_h.std(dim=0).clamp(min=1e-6)
    Xtr = (tr_h - mu) / sd
    Xva = (va_h - mu) / sd

    targets = {
        "need": (torch.tensor([r["utility"] for r in tr_rows], dtype=torch.float64),
                 torch.tensor([r["utility"] for r in va_rows], dtype=torch.float64)),
        "confidence": (torch.tensor([r["max_prob"] for r in tr_rows], dtype=torch.float64),
                       torch.tensor([r["max_prob"] for r in va_rows], dtype=torch.float64)),
    }

    out = {"mu": mu, "sd": sd, "bits": args.bits,
           "n_train": len(tr_rows), "n_val": len(va_rows), "probes": {}}
    report: dict[str, dict] = {}
    for name, (ytr, yva) in targets.items():
        best = None
        curve = []
        for lam in [float(x) for x in args.lams.split(",")]:
            w = ridge(Xtr, ytr, lam)
            pv = apply_w(Xva, w)
            r = pearson(pv, yva)
            rs = spearman(pv, yva)
            curve.append({"lam": lam, "val_pearson": r, "val_spearman": rs})
            if best is None or rs > best[1]:
                best = (lam, rs, w)
        lam, rs, w = best
        ptr = apply_w(Xtr, w)
        pva = apply_w(Xva, w)
        out["probes"][name] = w
        report[name] = {
            "target": name, "chosen_lambda": lam,
            "train_pearson": pearson(ptr, ytr), "train_spearman": spearman(ptr, ytr),
            "val_pearson": pearson(pva, yva), "val_spearman": spearman(pva, yva),
            "lambda_curve": curve,
            "selection_rule": "highest VALIDATION Spearman; test never consulted",
        }
        print("%-11s lam=%-8g  train rho=%.4f  VAL rho=%.4f  VAL r=%.4f"
              % (name, lam, report[name]["train_spearman"],
                 report[name]["val_spearman"], report[name]["val_pearson"]))

    # How well do the raw lens features predict need on val, for reference?
    for f in ("max_prob", "neg_entropy", "prob_margin", "logit_margin"):
        v = torch.tensor([r[f] for r in va_rows], dtype=torch.float64)
        u = targets["need"][1]
        report.setdefault("lens_baselines", {})[f] = {
            "val_spearman_negfeature_vs_need": spearman(-v, u),
            "val_pearson_negfeature_vs_need": pearson(-v, u),
        }
        print("  lens -%-13s vs need on VAL: rho=%+.4f  r=%+.4f"
              % (f, report["lens_baselines"][f]["val_spearman_negfeature_vs_need"],
                 report["lens_baselines"][f]["val_pearson_negfeature_vs_need"]))

    torch.save(out, REPO / args.out)
    (RES / "probe_report.json").write_text(json.dumps(report, indent=1) + "\n",
                                           encoding="utf-8")
    print("wrote %s and probe_report.json" % args.out)


if __name__ == "__main__":
    main()
