#!/usr/bin/env python3
"""Interventional group significance over all 96 groups of the four retired MLPs.

Definition, fixed before any arm was run (SGQR-v2 PREREGISTRATION section 9):

    U(j,g) = KL(dense || all-four-deleted) - KL(dense || only (j,g) active)

measured with **all four layers deleted**, which is the deployment-relevant
conditioning. CSGQR-v1 measured single-layer utility with only that layer
deleted; that answers a different question and would rank groups for a model
that does not exist at deployment.

Two things this deliberately does NOT do:

* It does not predict subset value. CSGQR-v1's N14 measured the additive proxy
  overstating a selection's true utility by up to 4x, worst for the most
  tailored selection. These scores are a **ranking signal only**; every budget
  point in this audit is measured by running the arm.
* It does not touch the evaluation split. Calibration is prompts 0..95;
  thresholds are set on 96..223; nothing is reported on anything but 224..509.

Rank stability is measured rather than assumed: the scores are recomputed on
bootstrap resamples of the calibration prompts and the Spearman correlation plus
top-k overlap against the full-sample ranking are recorded. A ranking that moves
between calibration subsets cannot support "significance beats random" as a
general claim, only as a claim about one calibration draw.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (CALIB, N_GROUPS, REPO, RETIRED, SPECS, TOTAL_GROUPS,      # noqa: E402
                  env_manifest, install, load_model, load_split,
                  reply_logprobs)
from multilayer import magnitude_scores                                     # noqa: E402
from stats import spearman as _spearman                                     # noqa: E402

import scripts.train.memory as mem                                          # noqa: E402
from envinfo import require_cuda                                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=96,
                    help="calibration prompts, taken from the calib split only")
    ap.add_argument("--bits", default="int4", choices=sorted(SPECS))
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--out", default="experiments/audit_sgqr_csgqr/results/significance_96.json")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(args.seed)

    model, tok = load_model(device)
    prompts = load_split(tok, CALIB, max_len=args.max_len)[:args.prompts]
    if not prompts:
        raise SystemExit("no usable prompts")
    inst = install(model, spec=SPECS[args.bits])

    print("interventional significance, 4-layer conditioning")
    print("  retired  %s   spec %s" % (list(RETIRED), SPECS[args.bits].name()))
    print("  groups   %d x %d = %d" % (len(RETIRED), N_GROUPS, TOTAL_GROUPS))
    print("  prompts  %d CALIBRATION prompts (eval starts at 224)" % len(prompts))
    print("  forwards %d per prompt\n" % (2 + TOTAL_GROUPS))

    NONE = {l: [] for l in RETIRED}
    per_prompt: list[list[float]] = []      # [prompt][flat group index]
    kl_deletes: list[float] = []
    t0 = time.time()
    for pi, p in enumerate(prompts):
        ids = p["ids"].to(device)
        b = p["boundary"]
        inst.use_dense()
        dense_lp = reply_logprobs(model, ids, b)
        pd = dense_lp.exp()

        def kl_of(lp):
            return float((pd * (dense_lp - lp)).sum(dim=-1).mean().item())

        inst.use_remnant()
        inst.set_levels(None)
        inst.set_allocation_by_level([NONE])
        kl_del = kl_of(reply_logprobs(model, ids, b))

        row: list[float] = []
        for l in RETIRED:
            for g in range(N_GROUPS):
                alloc = {x: ([g] if x == l else []) for x in RETIRED}
                inst.set_allocation_by_level([alloc])
                row.append(kl_del - kl_of(reply_logprobs(model, ids, b)))
        inst.use_dense()
        per_prompt.append(row)
        kl_deletes.append(kl_del)
        if (pi + 1) % 8 == 0:
            print("  %d/%d  %.0fs" % (pi + 1, len(prompts), time.time() - t0),
                  flush=True)

    U = torch.tensor(per_prompt, dtype=torch.float64)          # [P, 96]
    full = U.mean(dim=0)
    order_full = torch.argsort(full, descending=True).tolist()

    gen = torch.Generator().manual_seed(args.seed)
    rhos, ov = [], {24: [], 48: [], 10: [], 72: []}
    P = U.shape[0]
    for _ in range(args.bootstrap):
        idx = torch.randint(0, P, (P,), generator=gen)
        s = U.index_select(0, idx).mean(dim=0)
        rhos.append(_spearman(full, s))
        o = torch.argsort(s, descending=True).tolist()
        for k in ov:
            ov[k].append(len(set(o[:k]) & set(order_full[:k])) / k)

    rt = torch.tensor(rhos, dtype=torch.float64)
    inst.use_remnant()
    mag = magnitude_scores(inst.inner)
    inst.use_dense()

    scores = {str(l): full[i * N_GROUPS:(i + 1) * N_GROUPS].tolist()
              for i, l in enumerate(RETIRED)}
    payload = {
        "audit": "sgqr_csgqr_dual",
        "definition": ("U(j,g) = KL(dense||all-four-deleted) - "
                       "KL(dense||+(j,g)); larger is better"),
        "conditioning": "all four retired layers deleted",
        "ranking_signal_only": True,
        "retired_layers": list(RETIRED), "n_groups": N_GROUPS,
        "total_groups": TOTAL_GROUPS, "bits": args.bits,
        "spec": SPECS[args.bits].name(), "dataset": "chat_dev",
        "calibration_split": list(CALIB), "calibration_prompts": len(prompts),
        "held_out_split": [224, 510], "max_len": args.max_len, "seed": args.seed,
        "mean_kl_delete": float(sum(kl_deletes) / len(kl_deletes)),
        "scores": scores,
        "magnitude_scores": {str(l): mag[l] for l in RETIRED},
        "per_prompt_utility": U.tolist(),
        "rank_stability": {
            "bootstrap": args.bootstrap,
            "spearman_mean": float(rt.mean()),
            "spearman_p05": float(torch.quantile(rt, 0.05)),
            "spearman_p50": float(torch.quantile(rt, 0.50)),
            "topk_overlap_mean": {str(k): float(torch.tensor(v).mean()) for k, v in ov.items()},
            "topk_overlap_p05": {str(k): float(torch.quantile(torch.tensor(v), 0.05))
                                 for k, v in ov.items()},
        },
        "n_negative_groups": int((full < 0).sum()),
        "elapsed_s": round(time.time() - t0, 1),
        "environment": env_manifest(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print("\nmean KL(dense || all four deleted) = %.5f" % payload["mean_kl_delete"])
    for i, l in enumerate(RETIRED):
        v = full[i * N_GROUPS:(i + 1) * N_GROUPS]
        print("  L%-3d sum %8.5f  best %8.5f  worst %8.5f  negative %d/%d"
              % (l, float(v.sum()), float(v.max()), float(v.min()),
                 int((v < 0).sum()), N_GROUPS))
    print("\nsum of singleton utilities %.5f vs whole-remnant recovery (measured "
          "separately)" % float(full.sum()))
    print("rank stability over %d resamples:" % args.bootstrap)
    print("  Spearman vs full ranking  mean %.4f  p05 %.4f"
          % (payload["rank_stability"]["spearman_mean"],
             payload["rank_stability"]["spearman_p05"]))
    for k in (10, 24, 48, 72):
        print("  top-%-2d overlap            mean %.4f  p05 %.4f"
              % (k, payload["rank_stability"]["topk_overlap_mean"][str(k)],
                 payload["rank_stability"]["topk_overlap_p05"][str(k)]))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
