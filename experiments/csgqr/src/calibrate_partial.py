#!/usr/bin/env python3
"""Partial-execution gain: what a routed arm should be multiplying its output by.

The question this settles
-------------------------
Every routed arm in the Tier-1 sweep ran at ``alpha = 1``. That is defensible for
an always-on remnant, where ``alpha`` only has to undo quantization shrinkage and
measures ~0.998 at INT4. It is *not* obviously defensible at ``rho < 1``, because
the surviving model receives ``h + F_A`` where it would have received ``h + F``,
and ``F_A`` is a strict sub-sum. Nothing in the pipeline was rescaling it.

That matters because of a result the campaign could not otherwise explain. On the
singleton-marginal-utility metric, static top-6 captures 0.161 against random-6's
0.077 -- better by 2.1x. But measured at the model's output, static top-6 is
*worse* than random-6 (teacher-forced KL 0.371 vs 0.287, and mean quality 0.8195
vs 0.8282, both orderings agreeing). A ranking metric and the deployed behaviour
disagree in **sign**, which is exactly the shape of a missing normalisation.

The mechanism this tests: ranking by singleton marginal utility selects the
groups with the largest individual effect, so ``F_A`` for the top-k has a norm
close to ``||F||`` but points in a biased direction -- it overshoots along a
sub-basis. A random subset of the same size has a smaller norm and a more
isotropic direction. If that is what is happening, then the correct scalar gain
should (a) be furthest from 1 for the static selection, and (b) close or reverse
the static-vs-random gap once applied.

Two targets, because they answer different questions
----------------------------------------------------
``rows``   fit ``alpha F_A`` against ``sum_{g in A} F_g`` -- the same channels at
           full precision. "Reproduce faithfully what I chose to spend."
``whole``  fit ``alpha F_A`` against ``F`` -- the whole layer. "Use what I chose
           to best approximate the layer." This is the least-squares optimum for
           the perturbation the residual stream actually sees, and

               alpha = 1 + <F_A, F - F_A> / ||F_A||^2

           so it equals 1 exactly when the active groups are orthogonal to the
           inactive ones and rises toward ``1/rho`` as they become positively
           correlated. Its distance from 1 is therefore a *measurement* of
           cross-group correlation, reported here rather than assumed.

Metric
------
``rel_err = ||alpha F_A - F|| / ||F||`` -- the relative size of the error in the
residual-stream update, which is the quantity that propagates. Reported for
``alpha = 1`` and for both fitted gains, so the improvement attributable to
calibration is visible separately from the improvement attributable to selection.

Activations come from the real checkpoint at the real layer, taken at the
positions the model would be generating, so this is not a synthetic study.
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
from measure_layers import ARMS, DATASETS, load_prompts            # noqa: E402
from remnant import (QuantizedMlpRemnant, RemnantConfig,           # noqa: E402
                     calibrate_alpha, reference_group_output)
from significance import magnitude_ranking                         # noqa: E402

import scripts.train.memory as mem                                 # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
SEED = 20260823


@torch.no_grad()
def collect_mlp_inputs(model, tokenizer, layer: int, prompts: list[dict],
                       device) -> torch.Tensor:
    """Normalized MLP inputs at ``layer``, at reply positions only.

    The post-norm tensor is what Qwen3 feeds its MLP, so it is the correct input
    for a gain fitted on the MLP's own output. Restricted to reply positions
    because a gain averaged over prompt positions is fitted on tokens no arm ever
    has to produce.
    """
    grabbed: list[torch.Tensor] = []
    handle = model.model.layers[layer].post_attention_layernorm \
        .register_forward_hook(lambda _m, _i, out: grabbed.append(out.detach()))
    try:
        for p in prompts:
            grabbed.clear()
            model(p["ids"].to(device))
            out = grabbed[0][0, p["boundary"] - 1:-1]
            yield out
    finally:
        handle.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--fit-prompts", type=int, default=48,
                    help="prompts used to FIT the gains (calibration split)")
    ap.add_argument("--eval-prompts", type=int, default=48,
                    help="prompts used to EVALUATE, disjoint from the fit set")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--arm", default="int4", choices=sorted(ARMS))
    ap.add_argument("--ranking-from", default=None)
    ap.add_argument("--random-draws", type=int, default=16)
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--out",
                    default="experiments/csgqr/results/partial_gain_L20.json")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False

    total = args.fit_prompts + args.eval_prompts
    prompts = load_prompts(REPO / DATASETS[args.dataset], tokenizer, total,
                           args.max_len)
    if len(prompts) < total:
        raise SystemExit("needed " + str(total) + " prompts, loaded "
                         + str(len(prompts)))
    # Disjoint by prompt. A gain fitted and evaluated on the same positions would
    # report the in-sample optimum, which is not what a deployed arm gets.
    fit_prompts = prompts[:args.fit_prompts]
    eval_prompts = prompts[args.fit_prompts:total]

    donor = model.model.layers[args.layer].mlp
    spec = ARMS[args.arm]
    remnant = QuantizedMlpRemnant(
        donor, RemnantConfig(args.layer, args.n_groups, spec,
                             torch.bfloat16)).to(device)

    started = time.time()
    x_fit = torch.cat(list(collect_mlp_inputs(model, tokenizer, args.layer,
                                              fit_prompts, device)), dim=0)
    x_eval = torch.cat(list(collect_mlp_inputs(model, tokenizer, args.layer,
                                               eval_prompts, device)), dim=0)
    print("layer " + str(args.layer) + "   " + spec.name() + "   "
          + str(args.n_groups) + " groups")
    print("fit   " + str(len(fit_prompts)) + " prompts, "
          + str(x_fit.shape[0]) + " reply positions")
    print("eval  " + str(len(eval_prompts)) + " prompts, "
          + str(x_eval.shape[0]) + " reply positions   ("
          + format(time.time() - started, ".0f") + "s)\n")

    with torch.no_grad():
        f_eval = donor(x_eval).to(torch.float64)
    denom = f_eval.norm().item()

    ranking = list(range(args.n_groups))
    if args.ranking_from:
        src = json.loads((REPO / args.ranking_from).read_text(encoding="utf-8"))
        got = src.get("ranking") or src.get("significance_ranking")
        if got and len(got) == args.n_groups:
            ranking = list(got)
    mag = magnitude_ranking(remnant)

    def rel_err(groups: list[int], alpha: float) -> float:
        with torch.no_grad():
            out = remnant.forward_groups(x_eval, groups).to(torch.float64)
        return ((alpha * out - f_eval).norm() / denom).item()

    def norm_ratio(groups: list[int]) -> float:
        with torch.no_grad():
            out = remnant.forward_groups(x_eval, groups).to(torch.float64)
        return (out.norm() / denom).item()

    ks = sorted({max(1, int(round(f * args.n_groups)))
                 for f in (0.125, 0.25, 0.5, 0.75, 1.0)})
    gen = torch.Generator().manual_seed(SEED)

    rows = []
    print("{:>4s} {:>6s} {:<12s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "k", "rho", "selection", "|F_A|/|F|", "a_rows", "a_whole",
        "err_a=1", "err_rows", "err_whole"))
    for k in ks:
        selections: list[tuple[str, list[int]]] = [
            ("static_topk", sorted(ranking[:k])),
            ("magnitude", sorted(mag[:k])),
            ("contiguous", list(range(k))),
        ]
        for d in range(args.random_draws):
            perm = torch.randperm(args.n_groups, generator=gen)[:k].tolist()
            selections.append(("random_" + str(d), sorted(perm)))

        for name, groups in selections:
            a_rows = calibrate_alpha(remnant, donor, x_fit, subset=groups,
                                     against="rows")[0].item()
            a_whole = calibrate_alpha(remnant, donor, x_fit, subset=groups,
                                      against="whole")[0].item()
            entry = {
                "k": k, "rho": k / args.n_groups, "selection": name,
                "groups": groups,
                "norm_ratio": norm_ratio(groups),
                "alpha_rows": a_rows, "alpha_whole": a_whole,
                "rel_err_alpha1": rel_err(groups, 1.0),
                "rel_err_rows": rel_err(groups, a_rows),
                "rel_err_whole": rel_err(groups, a_whole),
            }
            rows.append(entry)
            if not name.startswith("random_") or name == "random_0":
                print("{:>4d} {:>6.3f} {:<12s} {:>8.4f} {:>9.4f} {:>9.4f} "
                      "{:>9.5f} {:>9.5f} {:>9.5f}".format(
                          k, entry["rho"], name, entry["norm_ratio"],
                          a_rows, a_whole, entry["rel_err_alpha1"],
                          entry["rel_err_rows"], entry["rel_err_whole"]))

    # -- the comparison the whole script exists for ------------------------
    print("\nstatic vs random at matched budget "
          "(mean over " + str(args.random_draws) + " random draws)")
    print("{:>4s} {:>6s} {:>26s} {:>26s} {:>26s}".format(
        "k", "rho", "alpha=1", "alpha=rows", "alpha=whole"))
    verdict = []
    for k in ks:
        st = next(r for r in rows if r["k"] == k and r["selection"] == "static_topk")
        rd = [r for r in rows if r["k"] == k and r["selection"].startswith("random_")]
        line = {"k": k, "rho": k / args.n_groups}
        for tag, field in (("alpha1", "rel_err_alpha1"),
                           ("rows", "rel_err_rows"),
                           ("whole", "rel_err_whole")):
            r_mean = sum(r[field] for r in rd) / len(rd)
            line[tag] = {"static": st[field], "random_mean": r_mean,
                         "static_better": bool(st[field] < r_mean)}
        line["alpha_whole_static"] = st["alpha_whole"]
        line["alpha_whole_random_mean"] = sum(r["alpha_whole"] for r in rd) / len(rd)
        verdict.append(line)
        print("{:>4d} {:>6.3f}".format(k, line["rho"])
              + "".join("   st {:>7.5f} rnd {:>7.5f} {:<4s}".format(
                  line[t]["static"], line[t]["random_mean"],
                  "ST" if line[t]["static_better"] else "RND")
                  for t in ("alpha1", "rows", "whole")))

    out = {
        "layer": args.layer, "dataset": args.dataset, "spec": spec.name(),
        "n_groups": args.n_groups,
        "fit_prompts": len(fit_prompts), "eval_prompts": len(eval_prompts),
        "fit_positions": int(x_fit.shape[0]),
        "eval_positions": int(x_eval.shape[0]),
        "splits_disjoint_by_prompt": True,
        "ranking_from": args.ranking_from,
        "significance_ranking": ranking,
        "magnitude_ranking": mag,
        "random_draws": args.random_draws,
        "metric": ("rel_err = ||alpha F_A - F|| / ||F|| over held-out reply "
                   "positions; F is the full-precision donor MLP output"),
        "rows": rows,
        "static_vs_random": verdict,
        "elapsed_s": round(time.time() - started, 1),
        "environment": collect(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
