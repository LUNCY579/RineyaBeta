#!/usr/bin/env python3
"""Does the gate signal vary WITHIN a prompt, or only between prompts?

A token-level gate earns its name only if its signal moves inside a sequence. A
signal that is really a per-prompt property would still score a positive overall
correlation with utility while being useless for the decision it is supposed to
make -- and the pooled correlation cannot tell the two apart.

So every signal is centred within its prompt before it is correlated with a
prompt-centred utility. What survives is the token-level component. The share of
utility variance that lives between prompts is reported alongside, because it
bounds how much of the pooled correlation could have come from the prompt level.

CPU only; reads the committed test trace and the fitted probes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import REPO                                            # noqa: E402
from stats import pearson, spearman                              # noqa: E402

RES = REPO / "experiments/audit_sgqr_csgqr/results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--bits", default="int4")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (RES / ("trace_%s_%s.jsonl" % (args.split, args.bits)))
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    hid = torch.load(RES / ("trace_%s_%s.hidden.pt" % (args.split, args.bits)),
                     weights_only=False)["hidden"].double()
    pr = torch.load(RES / "probes.pt", weights_only=False)
    X = (hid - pr["mu"]) / pr["sd"]
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)

    signals = {
        "probe_need": Xb @ pr["probes"]["need"],
        "probe_confidence": -(Xb @ pr["probes"]["confidence"]),
        "lens_max_prob": -torch.tensor([r["max_prob"] for r in rows], dtype=torch.float64),
        "lens_neg_entropy": -torch.tensor([r["neg_entropy"] for r in rows], dtype=torch.float64),
        "lens_prob_margin": -torch.tensor([r["prob_margin"] for r in rows], dtype=torch.float64),
        "lens_logit_margin": -torch.tensor([r["logit_margin"] for r in rows], dtype=torch.float64),
        "position_in_reply": -torch.tensor([r["pos"] for r in rows], dtype=torch.float64),
    }
    U = torch.tensor([r["utility"] for r in rows], dtype=torch.float64)
    D = torch.tensor([r["kl_deleted"] for r in rows], dtype=torch.float64)
    by: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["prompt"], []).append(i)

    def demean(v: torch.Tensor) -> torch.Tensor:
        o = v.clone()
        for idx in by.values():
            o[idx] = v[idx] - v[idx].mean()
        return o

    Ud, Dd = demean(U), demean(D)
    between = 1.0 - float(Ud.var()) / float(U.var())
    out = {"split": args.split, "bits": args.bits, "n_positions": len(rows),
           "n_prompts": len(by),
           "between_prompt_variance_share_utility": between,
           "between_prompt_variance_share_deletion_damage":
               1.0 - float(Dd.var()) / float(D.var()),
           "signals": {}}
    print("%s: %d positions, %d prompts" % (args.split, len(rows), len(by)))
    print("between-prompt share of utility variance: %.4f "
          "(so %.1f%% of what a token gate could exploit is WITHIN a prompt)"
          % (between, 100 * (1 - between)))
    print()
    print("%-20s %10s %12s %12s" % ("signal", "rho(pooled)", "rho(within)",
                                    "r(within)"))
    for k, s in signals.items():
        sd_ = demean(s)
        rec = {"spearman_pooled": spearman(s, U),
               "spearman_within_prompt": spearman(sd_, Ud),
               "pearson_within_prompt": pearson(sd_, Ud),
               "spearman_within_prompt_vs_deletion_damage": spearman(sd_, Dd)}
        out["signals"][k] = rec
        print("%-20s %+10.4f %+12.4f %+12.4f"
              % (k, rec["spearman_pooled"], rec["spearman_within_prompt"],
                 rec["pearson_within_prompt"]))
    (RES / ("within_prompt_signal_%s.json" % args.split)).write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("\nwrote within_prompt_signal_%s.json" % args.split)


if __name__ == "__main__":
    main()
