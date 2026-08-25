#!/usr/bin/env python3
"""How much of the gate's shortfall is the probe, and how much is the state?

The frontier shows a large gap between a confidence-gated arm and an oracle that
allocates by true per-token utility. Two very different explanations produce that
gap:

1. the **readout** is too weak -- a linear ridge on the layer-13 residual cannot
   express what is there; or
2. the **state** does not contain it -- the residual entering the first retired
   layer simply does not determine how much the remnant will be worth, three and
   twelve layers later.

They have opposite implications. If (1), a better probe is worth building. If (2),
no probe placed there can help and the gate should be abandoned at this placement.

This script bounds the answer without any GPU work, on the committed traces:

* an **in-sample** ridge with no effective regularization -- the best a linear
  readout can do on its own training data, an optimistic ceiling on the linear
  family;
* a small **MLP** (one hidden layer, tanh), trained on calib and selected on val
  -- if the non-linear model clears the linear one by much, the readout was the
  binding constraint;
* the **prompt-mean oracle** -- how much of the utility variance is between
  prompts rather than between tokens, which is the part a per-token gate cannot
  claim credit for;
* a **position-only** baseline, because "how far into the reply am I" is
  available for free and must be beaten before a state probe earns its cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import REPO                                             # noqa: E402
from stats import auroc, pearson, spearman                        # noqa: E402

RES = REPO / "experiments/audit_sgqr_csgqr/results"


def read(split: str, bits: str = "int4"):
    rows = [json.loads(l) for l in
            (RES / ("trace_%s_%s.jsonl" % (split, bits))).read_text(
                encoding="utf-8").splitlines() if l.strip()]
    hid = torch.load(RES / ("trace_%s_%s.hidden.pt" % (split, bits)),
                     weights_only=False)["hidden"].to(torch.float64)
    return rows, hid


def ridge(X, y, lam):
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(d + 1, dtype=X.dtype)
    A[d, d] -= lam
    return torch.linalg.solve(A, Xb.T @ y)


def apply_w(X, w):
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1) @ w


def capture(signal: torch.Tensor, util: torch.Tensor, frac: float) -> float:
    k = max(1, int(round(frac * signal.numel())))
    sel = torch.argsort(signal, descending=True)[:k]
    return float(util[sel].sum() / util.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="int4")
    ap.add_argument("--hidden", default="64,256")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tr, Htr = read("calib", args.bits)
    va, Hva = read("val", args.bits)
    te, Hte = read("test", args.bits)
    ytr = torch.tensor([r["utility"] for r in tr], dtype=torch.float64)
    yva = torch.tensor([r["utility"] for r in va], dtype=torch.float64)
    yte = torch.tensor([r["utility"] for r in te], dtype=torch.float64)
    mu, sd = Htr.mean(0), Htr.std(0).clamp(min=1e-6)
    Xtr, Xva, Xte = (Htr - mu) / sd, (Hva - mu) / sd, (Hte - mu) / sd

    out: dict = {"n_calib": len(tr), "n_val": len(va), "n_test": len(te),
                 "hidden_dim": int(Htr.shape[1]), "models": {}}

    def record(name, s_tr, s_va, s_te, extra=None):
        rec = {"train_spearman": spearman(s_tr, ytr),
               "val_spearman": spearman(s_va, yva),
               "test_spearman": spearman(s_te, yte),
               "test_pearson": pearson(s_te, yte),
               "test_auroc_delta0.1": auroc(s_te, (yte > 0.1).double()),
               "test_utility_captured": {str(f): capture(s_te, yte, f)
                                         for f in (0.1, 0.2, 0.3, 0.5)}}
        if extra:
            rec.update(extra)
        out["models"][name] = rec
        print("%-26s train rho %+.4f   val rho %+.4f   TEST rho %+.4f   "
              "TEST capture@20%% %.4f"
              % (name, rec["train_spearman"], rec["val_spearman"],
                 rec["test_spearman"], rec["test_utility_captured"]["0.2"]))

    # -- the oracle and the free baselines ---------------------------------
    record("oracle (true utility)", ytr, yva, yte)
    ptr = torch.tensor([r["pos"] for r in tr], dtype=torch.float64)
    pva = torch.tensor([r["pos"] for r in va], dtype=torch.float64)
    pte = torch.tensor([r["pos"] for r in te], dtype=torch.float64)
    record("position in reply (free)", -ptr, -pva, -pte)

    def prompt_mean(rows, y):
        by: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by.setdefault(r["prompt"], []).append(i)
        s = torch.zeros_like(y)
        for _, idx in by.items():
            s[idx] = y[idx].mean()
        return s
    record("prompt-mean ORACLE", prompt_mean(tr, ytr), prompt_mean(va, yva),
           prompt_mean(te, yte),
           {"note": "an oracle: uses the prompt's own mean utility. Bounds any "
                    "signal that varies only between prompts."})

    # -- linear family ------------------------------------------------------
    w_ins = ridge(Xtr, ytr, 1e-6)
    record("linear, IN-SAMPLE ceiling", apply_w(Xtr, w_ins), apply_w(Xva, w_ins),
           apply_w(Xte, w_ins),
           {"note": "lambda 1e-6: the best a linear readout can do on its own "
                    "training data. Its TEST column is what that optimism costs."})
    best = None
    for lam in (1, 10, 100, 1000, 1e4, 3e4, 1e5, 1e6):
        w = ridge(Xtr, ytr, lam)
        r = spearman(apply_w(Xva, w), yva)
        if best is None or r > best[0]:
            best = (r, lam, w)
    _, lam, w = best
    record("linear, val-selected ridge", apply_w(Xtr, w), apply_w(Xva, w),
           apply_w(Xte, w), {"chosen_lambda": lam})

    # -- non-linear family --------------------------------------------------
    Xtr32, Xva32, Xte32 = (Xtr.float(), Xva.float(), Xte.float())
    ytr32 = ytr.float()
    ymu, ysd = float(ytr32.mean()), float(ytr32.std())
    best_mlp = None
    for h in [int(v) for v in args.hidden.split(",")]:
        for wd in (1e-3, 1e-2, 1e-1):
            torch.manual_seed(args.seed)
            net = torch.nn.Sequential(
                torch.nn.Linear(Xtr32.shape[1], h), torch.nn.Tanh(),
                torch.nn.Linear(h, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
            tgt = ((ytr32 - ymu) / ysd).reshape(-1, 1)
            for _ in range(args.epochs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(Xtr32), tgt)
                loss.backward()
                opt.step()
            with torch.no_grad():
                sv = net(Xva32).squeeze(1).double()
            r = spearman(sv, yva)
            if best_mlp is None or r > best_mlp[0]:
                with torch.no_grad():
                    best_mlp = (r, h, wd,
                                net(Xtr32).squeeze(1).double(),
                                sv, net(Xte32).squeeze(1).double())
    r, h, wd, s_tr, s_va, s_te = best_mlp
    record("MLP, val-selected", s_tr, s_va, s_te,
           {"hidden_units": h, "weight_decay": wd, "epochs": args.epochs})

    lin = out["models"]["linear, val-selected ridge"]["test_spearman"]
    mlp = out["models"]["MLP, val-selected"]["test_spearman"]
    orc = 1.0
    out["verdict"] = {
        "linear_test_spearman": lin,
        "mlp_test_spearman": mlp,
        "nonlinear_gain": mlp - lin,
        "reading": ("the readout is the binding constraint" if mlp - lin > 0.10
                    else "the STATE is the binding constraint: a non-linear "
                         "readout of the same residual does not materially "
                         "improve on a linear one, so the information about "
                         "future remnant utility is largely absent from the "
                         "residual entering the first retired layer"),
        "fraction_of_oracle_rank_correlation": mlp / orc,
    }
    print("\n" + out["verdict"]["reading"])
    (RES / "probe_ceiling.json").write_text(json.dumps(out, indent=1) + "\n",
                                            encoding="utf-8")
    print("wrote probe_ceiling.json")


if __name__ == "__main__":
    main()
