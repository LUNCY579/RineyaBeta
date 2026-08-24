#!/usr/bin/env python3
"""Interventional group significance, conditioned on all four layers being deleted.

The conditioning is the whole point
-----------------------------------
With four retired layers there are 4 x 24 = 96 selectable groups, and the marginal
value of any one of them depends on what is active elsewhere. CSGQR-v1 measured
single-layer utility with only that layer deleted, which is the wrong
conditioning for a four-layer deployment: it asks "what does this group add to an
otherwise intact model?" when the deployed question is "starting from the
compressed model, what does turning this one group back on buy?"

So, exactly as preregistered:

    U(j,g) = KL(p_dense || p_all-four-deleted) - KL(p_dense || p_{+(j,g)})

measured with **all four layers deleted** and only group ``g`` of layer ``j``
switched on. Larger is better; a useless group scores about zero; a harmful one
scores negative.

Cost is ``2 + 4*n_groups`` forward passes per prompt -- one dense, one all-deleted,
and one per group. That is affordable because the remnants are installed **once**
and only the active group set is swapped between passes, so every group is
measured against exactly the same packed weights rather than against a freshly
quantized copy.

What this does NOT produce
--------------------------
A prediction of what any *subset* will achieve. CSGQR-v1 established that summing
singleton utilities overstates a selection's true value by up to 4x, and worst for
the most per-token selections (``NEGATIVE_RESULTS.md`` N14). These scores are a
**ranking signal only**. Every budget point in this campaign is measured by
running the arm.

Rank stability is reported alongside: the scores are recomputed on bootstrap
resamples of the calibration prompts, and Spearman correlation plus top-k overlap
are recorded. A ranking that moves wildly between calibration subsets is not
robust enough to deploy, and that has to be visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CSGQR_SRC = REPO / "experiments" / "csgqr" / "src"
for _p in (str(HERE), str(CSGQR_SRC), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envinfo import collect, require_cuda                          # noqa: E402
from measure_layers import DATASETS, load_prompts                  # noqa: E402
from quant import QuantSpec                                        # noqa: E402
from run_arms import MODEL, RETIRED, reply_logprobs                # noqa: E402
from multilayer import RetirementPlan, install_sgqr                # noqa: E402

import scripts.train.memory as mem                                 # noqa: E402


def spearman(a: list[float], b: list[float]) -> float:
    x = torch.tensor(a, dtype=torch.float64)
    y = torch.tensor(b, dtype=torch.float64)
    def rank(v):
        order = torch.argsort(v)
        r = torch.empty_like(v)
        r[order] = torch.arange(1, v.numel() + 1, dtype=torch.float64)
        return r
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    d = (rx.norm() * ry.norm()).item()
    return (rx @ ry).item() / d if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--prompts", type=int, default=64,
                    help="CALIBRATION prompts, taken from the start of the file "
                         "so they are disjoint from the held-out evaluation "
                         "slice, which begins at prompt 96")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8, 16))
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="resamples of the calibration prompts for rank stability")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--out",
                    default="experiments/sgqr_v2/results/significance_4layer.json")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False

    prompts = load_prompts(REPO / DATASETS[args.dataset], tokenizer,
                           args.prompts, args.max_len, skip=0)
    if not prompts:
        raise SystemExit("no usable prompts loaded")
    NG = args.n_groups
    spec = {4: QuantSpec(4, 128, False), 8: QuantSpec(8, 128, False),
            16: QuantSpec(16, 0, True)}[args.bits]

    print("interventional significance, conditioned on 4-layer deletion")
    print("  retired   " + str(list(RETIRED)) + "   " + spec.name())
    print("  groups    " + str(len(RETIRED)) + " x " + str(NG) + " = "
          + str(len(RETIRED) * NG))
    print("  prompts   " + str(len(prompts)) + " CALIBRATION prompts (held-out "
          "evaluation starts at 96)")
    print("  forwards  " + str(2 + len(RETIRED) * NG) + " per prompt\n")

    pristine = {l: model.model.layers[l].mlp for l in RETIRED}
    plan = RetirementPlan(layers=RETIRED, spec=spec, n_groups=NG)
    inst = install_sgqr(model, plan, allocation={l: [] for l in RETIRED},
                        keep_donor_resident=True)

    def use_remnants():
        for l in RETIRED:
            model.model.layers[l].mlp = inst.modules[l]

    def use_pristine():
        for l in RETIRED:
            model.model.layers[l].mlp = pristine[l]

    # per-prompt, per-group utility; kept per prompt so the bootstrap can
    # resample whole prompts rather than pooled tokens
    per_prompt: list[dict[str, list[float]]] = []
    t0 = time.time()
    for pi, p in enumerate(prompts):
        ids = p["ids"].to(device)
        b = p["boundary"]
        use_pristine()
        dense_lp = reply_logprobs(model, ids, b)
        pd = dense_lp.exp()

        def kl_against(lp):
            return float((pd * (dense_lp - lp)).sum(dim=-1).mean().item())

        use_remnants()
        inst.set_allocation({l: [] for l in RETIRED})
        kl_del = kl_against(reply_logprobs(model, ids, b))

        row: dict[str, list[float]] = {}
        for l in RETIRED:
            vals = []
            for g in range(NG):
                inst.set_allocation({x: ([g] if x == l else []) for x in RETIRED})
                vals.append(kl_del - kl_against(reply_logprobs(model, ids, b)))
            row[str(l)] = vals
        row["_kl_delete"] = [kl_del]
        per_prompt.append(row)
        use_pristine()
        if (pi + 1) % 8 == 0:
            print("  " + str(pi + 1) + "/" + str(len(prompts)) + "  "
                  + format(time.time() - t0, ".0f") + "s", flush=True)

    def mean_scores(idx):
        out = {}
        for l in RETIRED:
            out[l] = [sum(per_prompt[i][str(l)][g] for i in idx) / len(idx)
                      for g in range(NG)]
        return out

    full = mean_scores(list(range(len(per_prompt))))
    flat_full = [full[l][g] for l in RETIRED for g in range(NG)]

    # -- rank stability ----------------------------------------------------
    gen = torch.Generator().manual_seed(args.seed)
    rhos, overlaps24, overlaps48 = [], [], []
    order_full = sorted(range(len(flat_full)), key=lambda i: -flat_full[i])
    for _ in range(args.bootstrap):
        idx = torch.randint(0, len(per_prompt), (len(per_prompt),),
                            generator=gen).tolist()
        s = mean_scores(idx)
        flat = [s[l][g] for l in RETIRED for g in range(NG)]
        rhos.append(spearman(flat_full, flat))
        order = sorted(range(len(flat)), key=lambda i: -flat[i])
        overlaps24.append(len(set(order[:24]) & set(order_full[:24])) / 24)
        overlaps48.append(len(set(order[:48]) & set(order_full[:48])) / 48)

    rt = torch.tensor(rhos, dtype=torch.float64)
    o24 = torch.tensor(overlaps24, dtype=torch.float64)
    o48 = torch.tensor(overlaps48, dtype=torch.float64)
    kl_del_mean = sum(r["_kl_delete"][0] for r in per_prompt) / len(per_prompt)

    print("\nmean KL(dense || all four deleted) over calibration prompts: "
          + format(kl_del_mean, ".5f"))
    print("per-layer mean group utility:")
    for l in RETIRED:
        v = full[l]
        print("  L{:<3d} sum {:>8.5f}   best {:>8.5f}   worst {:>8.5f}   "
              "negative groups {:d}/{:d}".format(
                  l, sum(v), max(v), min(v), sum(1 for x in v if x < 0), NG))
    print("\nrank stability over " + str(args.bootstrap)
          + " prompt-bootstrap resamples:")
    print("  Spearman vs full-sample ranking : mean {:.4f}  p05 {:.4f}".format(
        rt.mean().item(), torch.quantile(rt, 0.05).item()))
    print("  top-24 overlap (rho=0.25)       : mean {:.4f}  p05 {:.4f}".format(
        o24.mean().item(), torch.quantile(o24, 0.05).item()))
    print("  top-48 overlap (rho=0.50)       : mean {:.4f}  p05 {:.4f}".format(
        o48.mean().item(), torch.quantile(o48, 0.05).item()))

    out = {
        "campaign": "sgqr_v2", "definition":
            "U(j,g) = KL(dense || all-four-deleted) - KL(dense || +(j,g)); "
            "measured with all four layers deleted; larger is better",
        "conditioning": "all four retired layers deleted",
        "ranking_signal_only": (
            "these scores rank groups; they do NOT predict subset value. "
            "CSGQR-v1 N14 measured the additive proxy overstating true subset "
            "utility by up to 4x. Every budget point is measured by running the arm."),
        "retired_layers": list(RETIRED), "n_groups": NG, "bits": args.bits,
        "spec": spec.name(), "dataset": args.dataset,
        "calibration_prompts": len(prompts), "max_len": args.max_len,
        "held_out_starts_at_prompt": 96,
        "mean_kl_delete": kl_del_mean,
        "scores": {str(l): full[l] for l in RETIRED},
        "rank_stability": {
            "bootstrap": args.bootstrap,
            "spearman_mean": rt.mean().item(),
            "spearman_p05": torch.quantile(rt, 0.05).item(),
            "top24_overlap_mean": o24.mean().item(),
            "top24_overlap_p05": torch.quantile(o24, 0.05).item(),
            "top48_overlap_mean": o48.mean().item(),
            "top48_overlap_p05": torch.quantile(o48, 0.05).item(),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "environment": collect(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
