#!/usr/bin/env python3
"""Fit the deployment probes: the group-utility router and the need gate.

``probe_utility.py`` established that both signals are learnable from the
pre-remnant residual -- need at AUROC 0.936 against a raw-lens 0.55-0.60, and
group utility winning 66-73% of the static-to-oracle headroom. But it only
*measured* that offline. It never produced an artifact that could be loaded into
a forward pass, so the dynamic half of CSGQR was analysed and never run.

This script produces the artifacts. Same estimators, same discipline, but the
output is a ``.pt`` a routed arm can consume.

Discipline, restated because it is what the numbers rest on
-----------------------------------------------------------
**Splits are by prompt, never by token.** Positions inside one prompt share a
context and a truncated prefix, so a token-level split leaks the answer across
the boundary and makes every probe look far better than it is.

**Three-way, not two.** The ridge/L2 strength has to be chosen somewhere, and
choosing it on the split it is then reported on inflates the number. ``val``
picks lambda; ``test`` is touched once, at the end.

**Closed form or deterministic optimiser.** Ridge is solved exactly; logistic
regression runs full-batch L-BFGS from a fixed zero initialisation. A reported
number is therefore a property of the data, not of a seed.

**The router is scored on utility captured, not on regression fit.** What matters
is not how well it predicts the numbers but how much real utility the groups it
selects actually deliver, so ``capture_at_k`` is the selection criterion for
lambda and the reported score. An R^2 would reward predicting the easy large
groups and say nothing about the top-k decision.

**Controls travel with the artifact.** The permuted-label fit and the
prompt-mean oracle are recorded in the saved metadata, not just printed, so a
loaded router carries the evidence about whether it was fitting signal or
capacity.

Why the probe is trained on the *quantized* donor's utilities
------------------------------------------------------------
The labels come from a trace generated at a specific bitwidth. Group utility
ranking need not survive requantization, so a router fitted on FP labels is the
wrong router to deploy at INT4. The saved metadata records the trace and its
spec, and ``load_group_router`` refuses a mismatch, so this cannot be got wrong
silently.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from envinfo import collect, require_cuda                          # noqa: E402
from measure_layers import DATASETS, load_prompts                  # noqa: E402
from probe_runtime import probe_cost, save_probe                   # noqa: E402
from probe_utility import (capture_at_k, logistic, ridge,          # noqa: E402
                           ridge_predict, standardize)
from stats import auprc, auroc, base_rate                          # noqa: E402
from trace_utility import capture_residual                         # noqa: E402

import scripts.train.memory as mem                                 # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--delta", type=float, default=0.1,
                    help="U > delta defines 'remnant needed'")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--lam", default="1.0,10.0,100.0,1000.0,10000.0")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--out-dir", default="artifacts/csgqr_probes")
    args = ap.parse_args()

    trace_dir = (Path(args.trace) if Path(args.trace).is_absolute()
                 else REPO / args.trace)
    rows = [json.loads(l) for l in (trace_dir / "traces.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((trace_dir / "trace_manifest.json")
                          .read_text(encoding="utf-8"))
    layer = manifest["layer"]
    n_groups = manifest["n_groups"]
    lams = [float(x) for x in args.lam.split(",")]

    # -- recover the residuals for exactly these positions ------------------
    require_cuda()
    mem.apply_cap(args.cap_mib)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
    model.config.use_cache = False
    hidden_size = int(model.config.hidden_size)
    d_ff = int(model.config.intermediate_size)
    prompts = load_prompts(REPO / DATASETS[manifest["dataset"]], tokenizer,
                           manifest["prompts"], 256)

    started = time.time()
    chunks = []
    for p in prompts:
        with torch.no_grad():
            r = capture_residual(model, layer, p["ids"].to("cuda:0"), p["boundary"])
        chunks.append(r.float().cpu())
    hidden = torch.cat(chunks, dim=0).to(torch.float64)
    del model
    torch.cuda.empty_cache()

    if hidden.shape[0] != len(rows):
        raise SystemExit("residual count " + str(hidden.shape[0])
                         + " does not match trace rows " + str(len(rows)))
    print("recovered " + str(hidden.shape[0]) + " residuals of width "
          + str(hidden.shape[1]) + " in "
          + format(time.time() - started, ".0f") + "s")

    util_g = torch.tensor([r["group_utility"] for r in rows], dtype=torch.float64)
    util = torch.tensor([r["utility"] for r in rows], dtype=torch.float64)
    prompt_idx = torch.tensor([r["prompt_index"] for r in rows])

    n_prompts = int(prompt_idx.max().item()) + 1
    cut_tr = int(round(args.train_frac * n_prompts))
    cut_val = cut_tr + int(round(args.val_frac * n_prompts))
    tr = prompt_idx < cut_tr
    va = (prompt_idx >= cut_tr) & (prompt_idx < cut_val)
    te = prompt_idx >= cut_val
    print("split by prompt: " + str(cut_tr) + "/" + str(cut_val - cut_tr) + "/"
          + str(n_prompts - cut_val) + " prompts   ("
          + str(int(tr.sum())) + "/" + str(int(va.sum())) + "/"
          + str(int(te.sum())) + " positions)")

    mu = hidden[tr].mean(dim=0, keepdim=True)
    sd = hidden[tr].std(dim=0, keepdim=True).clamp(min=1e-6)
    xt, xv, xs = standardize(hidden[tr], hidden[va], hidden[te])

    out_dir = REPO / args.out_dir
    tag = "L" + str(layer) + "_" + manifest["arm"] + "_g" + str(n_groups)
    common = {
        "trace": str(trace_dir.relative_to(REPO)).replace("\\", "/"),
        "layer": layer, "n_groups": n_groups, "hidden": hidden_size,
        "spec": manifest["spec"], "arm": manifest["arm"],
        "dataset": manifest["dataset"],
        "train_prompts": cut_tr, "val_prompts": cut_val - cut_tr,
        "test_prompts": n_prompts - cut_val,
        "train_positions": int(tr.sum()), "test_positions": int(te.sum()),
        "lambdas": lams,
        "environment": collect(),
    }

    # ======================================================================
    # A. Need gate
    # ======================================================================
    print("\nA. need gate  (U > " + str(args.delta) + ")")
    needed = (util > args.delta).double()
    print("   base rate: train " + format(base_rate(needed[tr]), ".4f")
          + "  val " + format(base_rate(needed[va]), ".4f")
          + "  test " + format(base_rate(needed[te]), ".4f"))

    def fit_logistic(x_tr, y_tr, x_va, y_va):
        best, best_a = None, -1e30
        for lam in lams:
            w = logistic(x_tr, y_tr, lam)
            s = (torch.cat([x_va, torch.ones(x_va.shape[0], 1,
                                             dtype=x_va.dtype)], 1) @ w).flatten()
            a = auroc(s, y_va)
            if a == a and a > best_a:
                best, best_a = lam, a
        return logistic(x_tr, y_tr, best), best, best_a

    w_need, lam_need, val_auroc = fit_logistic(xt, needed[tr], xv, needed[va])
    s_te = (torch.cat([xs, torch.ones(xs.shape[0], 1, dtype=xs.dtype)], 1)
            @ w_need).flatten()
    need_test = {"auroc": auroc(s_te, needed[te]),
                 "auprc": auprc(s_te, needed[te]),
                 "lam": lam_need, "val_auroc": val_auroc}
    print("   test AUROC " + format(need_test["auroc"], ".4f")
          + "  AUPRC " + format(need_test["auprc"], ".4f")
          + "   (lam " + format(lam_need, "g") + ", val "
          + format(val_auroc, ".4f") + ")")

    # Controls, saved with the artifact rather than merely printed.
    perm = torch.randperm(needed.numel(),
                          generator=torch.Generator().manual_seed(20260823))
    w_p, lam_p, _ = fit_logistic(xt, needed[perm][tr], xv, needed[perm][va])
    s_p = (torch.cat([xs, torch.ones(xs.shape[0], 1, dtype=xs.dtype)], 1)
           @ w_p).flatten()
    pmean = torch.zeros_like(util)
    for pi in range(n_prompts):
        sel = prompt_idx == pi
        if int(sel.sum()):
            pmean[sel] = needed[sel].mean()
    controls = {
        "permuted_labels_auroc": auroc(s_p, needed[perm][te]),
        "prompt_mean_oracle_auroc": auroc(pmean[te], needed[te]),
        "note": ("permuted must be ~0.5 or the pipeline leaks; the prompt-mean "
                 "oracle is the ceiling on any purely prompt-level signal, so "
                 "clearing it is what shows the probe resolves tokens"),
    }
    print("   controls: permuted " + format(controls["permuted_labels_auroc"], ".4f")
          + "  prompt-mean oracle " + format(controls["prompt_mean_oracle_auroc"], ".4f"))

    need_meta = {**common, "kind": "need", "delta": args.delta,
                 "base_rate_test": base_rate(needed[te]),
                 "test": need_test, "controls": controls,
                 "cost": probe_cost(hidden_size, 1, d_ff)}
    save_probe(out_dir / ("need_" + tag + ".pt"), kind="need",
               weight=w_need[:-1].reshape(1, -1).to(torch.float32),
               bias=w_need[-1].reshape(1).to(torch.float32),
               mu=mu.reshape(-1), sd=sd.reshape(-1), meta=need_meta)
    print("   wrote " + str((out_dir / ("need_" + tag + ".pt")).relative_to(REPO)))

    # ======================================================================
    # B. Group-utility router
    # ======================================================================
    print("\nB. group-utility router")
    ks = sorted({max(1, int(round(f * n_groups))) for f in (0.125, 0.25, 0.5, 0.75)})
    pick_k = ks[1] if len(ks) > 1 else ks[0]
    best_lam, best_score = None, -1e30
    for lam in lams:
        w = ridge(xt, util_g[tr], lam)
        s = capture_at_k(ridge_predict(w, xv), util_g[va], pick_k)
        if s > best_score:
            best_lam, best_score = lam, s
    w_router = ridge(xt, util_g[tr], best_lam)
    pred_te = ridge_predict(w_router, xs)
    true_te = util_g[te]

    static_rank = torch.argsort(util_g[tr].mean(dim=0), descending=True)
    g = torch.Generator().manual_seed(20260823)
    table = []
    print("   {:>4s} {:>6s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "k", "rho", "random", "static", "probe", "oracle", "headroom"))
    for k in ks:
        rand = sum(true_te[:, torch.randperm(n_groups, generator=g)[:k]]
                   .sum(1).mean().item() for _ in range(200)) / 200
        static = true_te[:, static_rank[:k]].sum(1).mean().item()
        probe = capture_at_k(pred_te, true_te, k)
        oracle = torch.topk(true_te, k=k, dim=1).values.sum(1).mean().item()
        won = (probe - static) / (oracle - static) if oracle > static else float("nan")
        table.append({"k": k, "rho": k / n_groups, "random": rand,
                      "static": static, "probe": probe, "oracle": oracle,
                      "headroom_fraction_won": won,
                      "probe_beats_static": bool(probe > static)})
        print("   {:>4d} {:>6.3f} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f} {:>8.1f}%"
              .format(k, k / n_groups, rand, static, probe, oracle, 100 * won))

    router_meta = {**common, "kind": "group_utility", "lam": best_lam,
                   "val_capture_at_k": best_score, "selection_k": pick_k,
                   "static_ranking_fitted_on_train": static_rank.tolist(),
                   "topk_capture": table,
                   "cost": probe_cost(hidden_size, n_groups, d_ff)}
    save_probe(out_dir / ("router_" + tag + ".pt"), kind="group_utility",
               weight=w_router[:-1].t().to(torch.float32),
               bias=w_router[-1].to(torch.float32),
               mu=mu.reshape(-1), sd=sd.reshape(-1), meta=router_meta)
    print("   wrote " + str((out_dir / ("router_" + tag + ".pt")).relative_to(REPO)))

    c = router_meta["cost"]
    print("\nmonitor cost per token, as a share of the MLP it routes:")
    print("   need gate    " + format(1e2 * need_meta["cost"]["share_of_mlp"], ".4f")
          + "%   (" + format(need_meta["cost"]["bytes_fp32"], ",") + " B)")
    print("   group router " + format(1e2 * c["share_of_mlp"], ".4f")
          + "%   (" + format(c["bytes_fp32"], ",") + " B)")
    print("   exact lens   " + format(1e2 * 32768 / (3 * d_ff), ".1f")
          + "%   -- the |V|/(3 d_ff) closed form, for contrast")

    summary = {"need": need_meta, "router": router_meta}
    dest = out_dir / ("probe_manifest_" + tag + ".json")
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
