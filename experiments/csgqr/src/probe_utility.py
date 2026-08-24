#!/usr/bin/env python3
"""Is the routing signal learnable from the pre-remnant state? (H1 revisited, H7)

Two questions that the trace analyses raise but cannot answer.

**Is there any signal for the "how much" gate?** The raw logit lens gives
AUROC ~0.55 for predicting whether the remnant is needed. That could mean the
information is absent from the pre-remnant state, or merely that the *raw* lens
is the wrong readout -- and those have opposite implications for the method. So a
probe is trained directly on the residual to predict remnant need. If a probe on
the full hidden state also fails, the information is genuinely not there and no
lens or calibration rescues it. If the probe succeeds, the lens was the problem.

**Is the dynamic-significance headroom capturable?** A static global ranking
leaves ~67% of the oracle's utility on the table at rho=0.25, and per-token
group orderings correlate only ~0.07 with the global one. That headroom is only
worth chasing if a router can predict per-group utility from the state. So a
probe is trained to predict all 24 group utilities, and is scored on the quantity
that matters: **the true utility captured by the groups it selects**, against the
static ranking below it and the oracle above it.

Discipline
----------
Splits are **by prompt**, never by token. Tokens from one prompt share a context
and a truncated prefix, so a token-level split would leak the answer across the
boundary and every probe here would look far better than it is.

Ridge regression is solved in closed form and logistic regression by full-batch
L-BFGS from a fixed initialisation, so a reported number is reproducible rather
than a function of an optimiser seed. Baselines are fitted the same way on the
four confidence features alone, so probe-versus-lens is a difference of input,
not of fitting procedure.
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

from envinfo import collect, require_cuda                        # noqa: E402
from measure_layers import DATASETS, load_prompts                # noqa: E402
from stats import auprc, auroc, base_rate                        # noqa: E402
from trace_utility import capture_residual                       # noqa: E402

import scripts.train.memory as mem                              # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
FEATURES = ("max_prob", "neg_entropy", "prob_margin", "logit_margin")


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge with a bias column. ``x``: [n, d], ``y``: [n, k]."""
    n, d = x.shape
    xb = torch.cat([x, torch.ones(n, 1, dtype=x.dtype)], dim=1)
    a = xb.t() @ xb
    # The bias is not penalised; penalising it would shrink the intercept toward
    # zero and bias every prediction on a target whose mean is far from zero.
    eye = torch.eye(d + 1, dtype=x.dtype)
    eye[d, d] = 0.0
    return torch.linalg.solve(a + lam * eye, xb.t() @ y)


def ridge_predict(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1) @ w


def logistic(x: torch.Tensor, y: torch.Tensor, lam: float,
             iters: int = 200) -> torch.Tensor:
    """L2-regularised logistic regression by full-batch L-BFGS.

    Deterministic: fixed zero initialisation, full batch, no shuffling. The bias
    column is unpenalised for the same reason as in :func:`ridge`.
    """
    n, d = x.shape
    xb = torch.cat([x, torch.ones(n, 1, dtype=x.dtype)], dim=1)
    w = torch.zeros(d + 1, 1, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")
    target = y.reshape(-1, 1).to(x.dtype)

    def closure():
        opt.zero_grad()
        logit = xb @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        loss = loss + lam * (w[:d] ** 2).sum() / n
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def standardize(train: torch.Tensor, *others: torch.Tensor):
    mu = train.mean(dim=0, keepdim=True)
    sd = train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return ((train - mu) / sd,) + tuple((o - mu) / sd for o in others)


def capture_at_k(pred: torch.Tensor, true: torch.Tensor, k: int) -> float:
    """True utility captured by the top-k groups of ``pred``, per token, averaged.

    This is the only honest score for a routing probe: what matters is not how
    well it predicts the numbers, but how much real utility the groups it picks
    actually deliver.
    """
    idx = torch.topk(pred, k=k, dim=1).indices
    return true.gather(1, idx).sum(dim=1).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--lam", default="1.0,10.0,100.0,1000.0,10000.0")
    ap.add_argument("--permute-control", action="store_true", default=True,
                    help="also fit on label-permuted data, which must give AUROC ~0.5")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace_dir = REPO / args.trace if not Path(args.trace).is_absolute() \
        else Path(args.trace)
    rows = [json.loads(l) for l in (trace_dir / "traces.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((trace_dir / "trace_manifest.json")
                          .read_text(encoding="utf-8"))
    layer = manifest["layer"]
    n_groups = manifest["n_groups"]

    # -- recover the residuals for exactly these positions ------------------
    require_cuda()
    mem.apply_cap(args.cap_mib)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(REPO / MODEL,
                                                 dtype=torch.bfloat16).to("cuda:0").eval()
    model.config.use_cache = False
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
                         + " does not match trace rows " + str(len(rows))
                         + "; the trace and this run disagree about the prompt set")
    print("recovered " + str(hidden.shape[0]) + " residuals of width "
          + str(hidden.shape[1]) + " in " + format(time.time() - started, ".0f") + "s")

    feats = torch.tensor([[r[f] for f in FEATURES] for r in rows],
                         dtype=torch.float64)
    util_g = torch.tensor([r["group_utility"] for r in rows], dtype=torch.float64)
    util = torch.tensor([r["utility"] for r in rows], dtype=torch.float64)
    prompt_idx = torch.tensor([r["prompt_index"] for r in rows])

    # -- three-way split, by prompt ----------------------------------------
    # Three ways, not two: the regularisation strength has to be chosen
    # somewhere, and choosing it on the split it is then reported on inflates
    # every number. val picks lambda; test is touched once.
    n_prompts = int(prompt_idx.max().item()) + 1
    cut_tr = int(round(args.train_frac * n_prompts))
    cut_val = cut_tr + int(round(args.val_frac * n_prompts))
    train_mask = prompt_idx < cut_tr
    val_mask = (prompt_idx >= cut_tr) & (prompt_idx < cut_val)
    test_mask = prompt_idx >= cut_val
    n_tr, n_va, n_te = (int(train_mask.sum()), int(val_mask.sum()),
                        int(test_mask.sum()))
    print("split by prompt: " + str(cut_tr) + " train / "
          + str(cut_val - cut_tr) + " val / " + str(n_prompts - cut_val)
          + " test prompts   (" + str(n_tr) + " / " + str(n_va) + " / "
          + str(n_te) + " positions)")
    print("NOTE the probe has " + str(hidden.shape[1] + 1)
          + " parameters against " + str(n_tr) + " training positions, so it is "
          "overparameterised; the permutation control below is what tells us "
          "whether it is fitting signal or capacity.")

    needed = (util > args.delta).double()
    print("label P(U > " + str(args.delta) + "): train "
          + format(base_rate(needed[train_mask]), ".4f") + "  val "
          + format(base_rate(needed[val_mask]), ".4f") + "  test "
          + format(base_rate(needed[test_mask]), ".4f"))
    print()

    lams = [float(x) for x in args.lam.split(",")]
    results: dict = {}

    # ======================================================================
    # A. Predicting whether the remnant is needed
    # ======================================================================
    print("A. predicting 'remnant needed' (U > " + str(args.delta)
          + ") on held-out prompts")
    print("   {:<28s} {:>9s} {:>9s}".format("predictor", "AUROC", "AUPRC"))

    # The raw lens features, used directly (no fitting) -- the deployed gate.
    raw = {}
    for i, f in enumerate(FEATURES):
        s = -feats[test_mask][:, i]        # low confidence should mean needed
        raw[f] = {"auroc": auroc(s, needed[test_mask]),
                  "auprc": auprc(s, needed[test_mask])}
        print("   {:<28s} {:>9.4f} {:>9.4f}".format(
            "raw " + f, raw[f]["auroc"], raw[f]["auprc"]))

    def fit_score(x_all: torch.Tensor, labels: torch.Tensor) -> dict:
        """Pick lambda on val, report on test. Never the other way round."""
        xt, xv, xs = standardize(x_all[train_mask], x_all[val_mask],
                                 x_all[test_mask])
        best_lam, best_val = None, -1e30
        for lam in lams:
            w = logistic(xt, labels[train_mask], lam)
            sv = (torch.cat([xv, torch.ones(xv.shape[0], 1,
                                            dtype=xv.dtype)], dim=1) @ w).flatten()
            a = auroc(sv, labels[val_mask])
            if a == a and a > best_val:
                best_lam, best_val = lam, a
        w = logistic(xt, labels[train_mask], best_lam)
        st = (torch.cat([xs, torch.ones(xs.shape[0], 1,
                                        dtype=xs.dtype)], dim=1) @ w).flatten()
        return {"lam": best_lam, "val_auroc": best_val,
                "auroc": auroc(st, labels[test_mask]),
                "auprc": auprc(st, labels[test_mask])}

    fitted: dict = {}
    for name, x_all in (("logistic on 4 lens features", feats),
                        ("logistic on hidden state", hidden)):
        best = fit_score(x_all, needed)
        fitted[name] = best
        print("   {:<28s} {:>9.4f} {:>9.4f}   (lam {:g}, val {:.4f})".format(
            name, best["auroc"], best["auprc"], best["lam"], best["val_auroc"]))

    # Controls. A probe with 1025 parameters and a few hundred training rows can
    # look good for the wrong reason, so three things are checked:
    #
    #   permuted   labels shuffled across all tokens. Any AUROC materially above
    #              0.5 here means the pipeline leaks, full stop.
    #   position   the reply position alone. Utility could simply decline along a
    #              reply, in which case the probe learned a counter, not a state.
    #   prompt-mean each token gets its own prompt's mean label. This is the
    #              ceiling on any purely prompt-level signal, so if the probe
    #              only matches it, the probe is not resolving tokens.
    controls: dict = {}
    perm = torch.randperm(needed.numel(),
                          generator=torch.Generator().manual_seed(20260823))
    controls["permuted_labels"] = fit_score(hidden, needed[perm])
    print("   {:<28s} {:>9.4f} {:>9.4f}   <- must be ~0.5".format(
        "permuted-label control", controls["permuted_labels"]["auroc"],
        controls["permuted_labels"]["auprc"]))

    pos = torch.tensor([[float(r["position"])] for r in rows], dtype=torch.float64)
    controls["position_only"] = fit_score(pos, needed)
    print("   {:<28s} {:>9.4f} {:>9.4f}".format(
        "reply position only", controls["position_only"]["auroc"],
        controls["position_only"]["auprc"]))

    pmean = torch.zeros_like(util)
    for pidx in range(n_prompts):
        sel = prompt_idx == pidx
        if int(sel.sum()):
            pmean[sel] = needed[sel].mean()
    controls["prompt_mean_oracle"] = {
        "auroc": auroc(pmean[test_mask], needed[test_mask]),
        "auprc": auprc(pmean[test_mask], needed[test_mask]),
        "note": "an oracle: uses the test prompt's own mean label",
    }
    print("   {:<28s} {:>9.4f} {:>9.4f}   <- prompt-level ceiling (oracle)".format(
        "prompt-mean oracle", controls["prompt_mean_oracle"]["auroc"],
        controls["prompt_mean_oracle"]["auprc"]))

    results["remnant_needed"] = {
        "delta": args.delta,
        "base_rate_train": base_rate(needed[train_mask]),
        "base_rate_val": base_rate(needed[val_mask]),
        "base_rate_test": base_rate(needed[test_mask]),
        "raw_lens_features": raw,
        "fitted": fitted,
        "controls": controls,
    }
    print()

    # ======================================================================
    # B. Predicting which groups help -- the number that decides dynamic routing
    # ======================================================================
    print("B. utility captured by top-k groups on held-out prompts")
    static_rank = torch.argsort(util_g[train_mask].mean(dim=0),
                                descending=True)          # fitted on TRAIN only
    ks = sorted({max(1, int(round(f * n_groups))) for f in (0.125, 0.25, 0.5, 0.75)})
    true_te = util_g[test_mask]

    probes: dict[str, torch.Tensor] = {}
    val_g = util_g[val_mask]
    for name, x_all in (("hidden", hidden), ("lens_features", feats),
                        ("permuted", hidden)):
        target = util_g[train_mask]
        if name == "permuted":
            # Rows shuffled: destroys the state-to-utility correspondence while
            # keeping both marginal distributions intact.
            shuffle = torch.randperm(int(train_mask.sum()),
                                     generator=torch.Generator().manual_seed(7))
            target = target[shuffle]
        xt, xv, xs = standardize(x_all[train_mask], x_all[val_mask],
                                 x_all[test_mask])
        best_lam, best_val = None, -1e30
        for lam in lams:
            w = ridge(xt, target, lam)
            s = capture_at_k(ridge_predict(w, xv), val_g, ks[1])
            if s > best_val:
                best_lam, best_val = lam, s
        w = ridge(xt, target, best_lam)
        probes[name] = ridge_predict(w, xs)
        results.setdefault("probe_lambda", {})[name] = best_lam

    table = []
    print("   {:>4s} {:>6s} {:>9s} {:>9s} {:>9s} {:>10s} {:>9s} {:>9s} {:>9s}".format(
        "k", "rho", "random", "permuted", "static", "probe(lens)", "probe(h)",
        "oracle", "headroom"))
    g = torch.Generator().manual_seed(20260823)
    for k in ks:
        rand_scores = []
        for _ in range(200):
            perm = torch.randperm(n_groups, generator=g)[:k]
            rand_scores.append(true_te[:, perm].sum(dim=1).mean().item())
        rand = sum(rand_scores) / len(rand_scores)
        static = true_te[:, static_rank[:k]].sum(dim=1).mean().item()
        probe = capture_at_k(probes["hidden"], true_te, k)
        probe_f = capture_at_k(probes["lens_features"], true_te, k)
        probe_p = capture_at_k(probes["permuted"], true_te, k)
        oracle = torch.topk(true_te, k=k, dim=1).values.sum(dim=1).mean().item()
        won = ((probe - static) / (oracle - static)) if oracle > static else float("nan")
        table.append({"k": k, "rho": k / n_groups, "random": rand,
                      "permuted_probe": probe_p,
                      "static": static, "probe_hidden": probe,
                      "probe_lens_features": probe_f, "oracle": oracle,
                      "headroom_fraction_won": won,
                      "probe_beats_static": bool(probe > static),
                      "probe_beats_permuted": bool(probe > probe_p)})
        print("   {:>4d} {:>6.3f} {:>9.5f} {:>9.5f} {:>9.5f} {:>10.5f} {:>9.5f}"
              " {:>9.5f} {:>8.1f}%".format(
                  k, k / n_groups, rand, probe_p, static, probe_f, probe,
                  oracle, 100 * won))
    results["topk_capture"] = table

    out = {
        "trace": str(trace_dir.relative_to(REPO)).replace("\\", "/"),
        "layer": layer, "n_groups": n_groups,
        "dataset": manifest["dataset"],
        "n_positions": len(rows), "n_prompts": n_prompts,
        "train_prompts": cut_tr, "val_prompts": cut_val - cut_tr,
        "test_prompts": n_prompts - cut_val,
        "train_positions": n_tr, "val_positions": n_va, "test_positions": n_te,
        "hidden_width": int(hidden.shape[1]),
        "lambdas": lams,
        "static_ranking_fitted_on_train": static_rank.tolist(),
        **results,
        "environment": collect(),
    }
    dest = REPO / (args.out or (str(trace_dir.relative_to(REPO)).replace("\\", "/")
                                + "/probe_analysis.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
