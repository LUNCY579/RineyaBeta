#!/usr/bin/env python3
"""Gate D: the threshold frontier, simulated from a trace and then validated.

Why simulate
------------
A full quality run per threshold pair is ~10 minutes; a 6x6 grid across four
confidence features is a day. The trace already contains, per token, the
confidence features and the utility of every group, so a policy's *routing
behaviour* -- active fraction, invocation rate, which tokens get skipped, how
much measured utility those skips forgo -- can be replayed exactly without
touching the GPU. That makes an interior frontier cheap to find.

What simulation gets exactly, and what it approximates
------------------------------------------------------
**Exact:** every routing quantity. The policy is a deterministic function of the
confidence features, which the trace stores per token, so ``active_fraction``,
``skip_rate``, ``k_t`` and the selected group sets are reproduced exactly rather
than estimated.

**Approximate:** the utility. Each per-group utility ``U_{t,g}`` in the trace was
measured with *every* position running that same configuration. A policy that
activates group ``g`` at token ``t`` but not at token ``t'`` produces slightly
different hidden states at ``t``, because ``t'``'s state reaches ``t`` through
attention in the layers above ``j``. The simulated utility therefore assumes
per-token independence, which is not exact.

It also assumes group-level additivity: the utility of a set is taken as the sum
of its members' utilities. ``analyze_significance.py`` measures the interaction
term directly, so the size of that error is known rather than hoped about.

Both approximations are why ``--validate-points`` exists: it prints the exact
``run_eval.py`` invocations for the selected frontier points, so the simulated
frontier is confirmed by real generation before anything is claimed from it. A
simulated frontier on its own is a hypothesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from router import RoutingPolicy                                  # noqa: E402
from significance import GroupSignificance                        # noqa: E402
from stats import cluster_bootstrap                               # noqa: E402

FEATURES = ("max_prob", "neg_entropy", "prob_margin")


def simulate(policy: RoutingPolicy, conf: list[float], util: torch.Tensor,
             *, dynamic: bool) -> dict:
    """Replay a policy over the trace.

    ``dynamic=True`` selects per token using that token's own measured utilities
    -- an *oracle* dynamic router, since a real one would have to predict them.
    It bounds what any learned per-state router could achieve.
    """
    n_tokens, n_groups = util.shape
    executed = 0
    invoked = 0
    full = 0
    captured = torch.zeros(n_tokens, dtype=torch.float64)
    budgets: list[int] = []

    for t in range(n_tokens):
        u = util[t] if dynamic else None
        groups = policy.select(conf[t], utility=u, token_index=t)
        k = len(groups)
        budgets.append(k)
        executed += k
        invoked += int(k > 0)
        full += int(k == n_groups)
        if groups:
            captured[t] = util[t, torch.tensor(groups, dtype=torch.long)].sum()

    return {
        "active_fraction": executed / (n_tokens * n_groups),
        "invocation_rate": invoked / n_tokens,
        "full_invocation_rate": full / n_tokens,
        "skip_rate": 1.0 - invoked / n_tokens,
        "mean_groups_per_token": executed / n_tokens,
        "captured": captured,
        "budgets": budgets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--feature", default="max_prob", choices=FEATURES)
    ap.add_argument("--gamma", default="1.0")
    ap.add_argument("--tau-skip", default="0.5,0.7,0.8,0.9,0.95,0.99,1.01")
    ap.add_argument("--tau-full", default="0.0,0.2,0.4,0.6,0.8")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--validate-points", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace_dir = REPO / args.trace if not Path(args.trace).is_absolute() \
        else Path(args.trace)
    rows = [json.loads(l) for l in (trace_dir / "traces.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((trace_dir / "trace_manifest.json")
                          .read_text(encoding="utf-8"))
    n_groups = manifest["n_groups"]
    clusters = [r["prompt_index"] for r in rows]
    conf = [float(r[args.feature]) for r in rows]
    util = torch.tensor([r["group_utility"] for r in rows], dtype=torch.float64)
    whole = torch.tensor([r["utility"] for r in rows], dtype=torch.float64)

    sig = GroupSignificance(n_groups)
    sig.update(util)
    ranking = sig.ranking()

    # The reference points: nothing, and everything.
    all_pol = RoutingPolicy(mode="all", n_groups=n_groups, ranking=ranking)
    none_sim = {"active_fraction": 0.0, "captured": torch.zeros(len(rows),
                                                               dtype=torch.float64)}
    all_sim = simulate(all_pol, conf, util, dynamic=False)
    full_capture = all_sim["captured"].mean().item()
    whole_mean = whole.mean().item()

    print("trace   layer " + str(manifest["layer"]) + "  " + manifest["arm"]
          + "  " + str(len(rows)) + " positions, "
          + str(len(set(clusters))) + " prompts")
    print("feature " + args.feature)
    print("reference: all groups capture " + format(full_capture, ".5f")
          + " (sum of group utilities);  measured whole-remnant utility "
          + format(whole_mean, ".5f"))
    print("           additivity gap " + format(full_capture - whole_mean, "+.5f")
          + "  <- simulated utility inherits this error\n")

    gammas = [float(x) for x in args.gamma.split(",")]
    skips = [float(x) for x in args.tau_skip.split(",")]
    fulls = [float(x) for x in args.tau_full.split(",")]
    needed = (whole > args.delta)

    grid: list[dict] = []
    for gamma in gammas:
        for ts in skips:
            for tf in fulls:
                if not tf < ts:
                    continue
                for selector, dynamic in (("static_topk", False),
                                          ("significance", True),
                                          ("random", False)):
                    pol = RoutingPolicy(mode="three_level", feature=args.feature,
                                        tau_skip=ts, tau_full=tf, gamma=gamma,
                                        selector=selector, n_groups=n_groups,
                                        ranking=ranking)
                    sim = simulate(pol, conf, util, dynamic=dynamic)
                    cap = sim["captured"]
                    ci = cluster_bootstrap(cap, clusters,
                                           lambda v: v.mean().item(),
                                           n_boot=args.n_boot)
                    skipped = torch.tensor([b == 0 for b in sim["budgets"]])
                    false_skips = int((skipped & needed).sum())
                    grid.append({
                        "gamma": gamma, "tau_skip": ts, "tau_full": tf,
                        "selector": selector, "oracle_dynamic": dynamic,
                        "active_fraction": sim["active_fraction"],
                        "invocation_rate": sim["invocation_rate"],
                        "full_invocation_rate": sim["full_invocation_rate"],
                        "skip_rate": sim["skip_rate"],
                        "mean_groups_per_token": sim["mean_groups_per_token"],
                        "captured_utility": ci["point"],
                        "captured_ci_lo": ci["lo"], "captured_ci_hi": ci["hi"],
                        "utility_retained_fraction": (ci["point"] / full_capture
                                                      if full_capture else float("nan")),
                        "false_skip_count": false_skips,
                        "false_skip_rate_given_needed":
                            false_skips / max(int(needed.sum()), 1),
                    })

    interior = [g for g in grid if 0.0 < g["active_fraction"] < 1.0]
    print("grid points: " + str(len(grid)) + "   interior (0 < rho < 1): "
          + str(len(interior)))

    # The frontier: for each active fraction, the point retaining the most utility.
    by_rho: dict[str, dict] = {}
    for g in grid:
        key = format(round(g["active_fraction"], 3), ".3f")
        if key not in by_rho or g["utility_retained_fraction"] > \
                by_rho[key]["utility_retained_fraction"]:
            by_rho[key] = g
    frontier = sorted(by_rho.values(), key=lambda g: g["active_fraction"])

    print("\nfrontier (best utility retention at each active fraction):")
    print("  {:>7s} {:>10s} {:>10s} {:>12s} {:>16s} {:>14s}".format(
        "rho", "skip rate", "retained", "selector", "P(skip|needed)", "oracle?"))
    for g in frontier:
        print("  {:>7.3f} {:>10.3f} {:>10.4f} {:>12s} {:>16.4f} {:>14s}".format(
            g["active_fraction"], g["skip_rate"], g["utility_retained_fraction"],
            g["selector"], g["false_skip_rate_given_needed"],
            "yes" if g["oracle_dynamic"] else ""))

    # Matched-compute comparison: at the same rho, does significance beat random?
    print("\nmatched-compute comparison at equal active fraction:")
    print("  {:>7s} {:>12s} {:>12s} {:>12s} {:>10s}".format(
        "rho", "static", "random", "oracle dyn", "static>rand"))
    buckets: dict[str, dict[str, float]] = {}
    for g in grid:
        key = format(round(g["active_fraction"], 2), ".2f")
        b = buckets.setdefault(key, {})
        name = "oracle" if g["oracle_dynamic"] else g["selector"]
        prev = b.get(name)
        if prev is None or g["utility_retained_fraction"] > prev:
            b[name] = g["utility_retained_fraction"]
    matched = []
    for key in sorted(buckets, key=float):
        b = buckets[key]
        if "static_topk" in b and "random" in b:
            wins = b["static_topk"] > b["random"]
            matched.append({"active_fraction": float(key),
                            "static": b["static_topk"], "random": b["random"],
                            "oracle": b.get("oracle"),
                            "static_beats_random": bool(wins)})
            print("  {:>7s} {:>12.4f} {:>12.4f} {:>12s} {:>10s}".format(
                key, b["static_topk"], b["random"],
                format(b["oracle"], ".4f") if b.get("oracle") is not None else "-",
                "yes" if wins else "NO"))

    picks = sorted(interior, key=lambda g: -g["utility_retained_fraction"]
                   )[:args.validate_points] if interior else []
    if picks:
        print("\nvalidate these points with real generation:")
        for g in picks:
            print("  .venv/Scripts/python.exe experiments/csgqr/src/run_eval.py"
                  " --arm b10-csgqr --layer " + str(manifest["layer"])
                  + " --dataset " + manifest["dataset"]
                  + " --feature " + args.feature
                  + " --tau-skip " + format(g["tau_skip"], ".3f")
                  + " --tau-full " + format(g["tau_full"], ".3f")
                  + " --gamma " + format(g["gamma"], ".2f")
                  + "   # simulated rho=" + format(g["active_fraction"], ".3f"))

    out = {
        "trace": str(trace_dir.relative_to(REPO)).replace("\\", "/"),
        "layer": manifest["layer"], "arm": manifest["arm"],
        "feature": args.feature, "n_groups": n_groups,
        "n_positions": len(rows), "n_prompts": len(set(clusters)),
        "delta": args.delta,
        "ranking": ranking,
        "full_capture_sum_of_groups": full_capture,
        "measured_whole_remnant_utility": whole_mean,
        "additivity_gap": full_capture - whole_mean,
        "simulation_caveat":
            "routing quantities are exact; utilities assume per-token "
            "independence and group additivity, so frontier points must be "
            "confirmed by run_eval.py",
        "grid": grid,
        "interior_point_count": len(interior),
        "frontier": frontier,
        "matched_compute": matched,
        "validate_points": picks,
    }
    dest = REPO / (args.out or (str(trace_dir.relative_to(REPO)).replace("\\", "/")
                                + "/threshold_sweep_" + args.feature + ".json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
