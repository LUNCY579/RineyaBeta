#!/usr/bin/env python3
"""Matched-compute SGQR vs CSGQR frontier, with the required factorial ablation.

What is controlled
------------------
Every arm at a budget point executes **exactly the same number of group-token
products** -- not approximately, exactly, by construction
(:func:`core.allocate_exact`). Every arm runs through the same module, the same
kernels and the same allocator. Every arm sees the same prompts, the same
positions, the same gold tokens and the same packed remnant weights, because the
remnant is quantized once for the whole sweep and only the level assignment
changes. The gate signal is captured from the residual entering the first
retired layer, which no arm modifies, so all arms are gated on literally the
same numbers.

That leaves exactly two axes free, which is the factorial the brief asks for:

    demand   flat (SGQR: same budget every token)  vs
             confidence-derived (CSGQR: budget varies per token)
    order    interventional significance  vs  random  vs  magnitude

    remnant only                      flat   + random
    remnant + significance            flat   + significance      == SGQR
    remnant + confidence              gated  + random
    remnant + significance + confidence  gated + significance    == CSGQR

Controls that bound the answer from both sides
----------------------------------------------
``randtok-*``  takes CSGQR's own realized per-token budget vector and PERMUTES it
               across tokens. Same multiset of budgets, same total, same order:
               the only difference is which token got which budget. This is the
               sharpest possible test of "does confidence-aware allocation beat
               random allocation at equal compute", and it is strictly tighter
               than the historical random-token gate, which matched only the
               expected activation.
``oracle-*``   allocates by the TRUE per-token remnant utility, measured on the
               same split. It cannot be deployed; it is the ceiling on what any
               token-level gate could achieve, and it is the number that decides
               whether a better confidence signal is worth looking for.

Splits: budgets and probes come from calib/val; this script reports on whichever
split is passed and the test split is intended to be run once.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (CALIB, N_GROUPS, REPO, RETIRED, SPECS, TEST, TOTAL_GROUPS,   # noqa: E402
                  VAL, alloc_from_order, allocate_exact, binary_budgets,
                  env_manifest, global_order, install, lens_features,
                  load_model, load_split, make_head, position_metrics,
                  random_order, ResidualTap)
from stats import (cluster_bootstrap_mean, cluster_bootstrap_mean_pair,        # noqa: E402
                   holm_bonferroni)

import scripts.train.memory as mem                                             # noqa: E402
from envinfo import require_cuda                                               # noqa: E402

SPLITS = {"calib": CALIB, "val": VAL, "test": TEST}
RES = REPO / "experiments/audit_sgqr_csgqr/results"


@dataclass
class Arm:
    key: str
    demand: str          # flat | probe_need | conf_probe | lens_maxprob | oracle | perm:<arm>:<seed>
    order: str           # significance | random_s<seed> | magnitude | reversed
    mode: str            # graded | binary | fixedfull | none | dense
    budget: float
    label: str = ""


def build_arms(budgets: list[float], seeds: list[int],
               perm_for: tuple[str, ...] = ("probe",)) -> list[Arm]:
    """``perm_for`` names which gated arms get a permuted-budget twin.

    A permuted twin holds the arm's own multiset of per-token budgets and its
    total fixed, and only re-assigns which token gets which. Without one for a
    given gate, a win over the fixed-budget arm cannot be attributed to the
    gate's ranking rather than to the mere SHAPE of the budget distribution it
    induces -- a distinction the historical random-token control does not make.
    """
    arms = [Arm("dense", "flat", "significance", "dense", 1.0, "dense reference"),
            Arm("off-delete4", "flat", "significance", "none", 0.0,
                "four layers deleted, no remnant"),
            Arm("on-int4-full", "flat", "significance", "fixedfull", 1.0,
                "always-on INT4 remnant, all 96 groups")]
    for b in budgets:
        t = format(b, ".2f")
        arms.append(Arm("sgqr-sig-" + t, "flat", "significance", "graded", b,
                        "SGQR: fixed budget, interventional significance"))
        for s in seeds:
            arms.append(Arm("sgqr-rand-s%d-" % s + t, "flat", "random_s%d" % s,
                            "graded", b, "fixed budget, random group order"))
        arms.append(Arm("sgqr-mag-" + t, "flat", "magnitude", "graded", b,
                        "fixed budget, weight-magnitude order"))
        arms.append(Arm("sgqr-rev-" + t, "flat", "reversed", "graded", b,
                        "fixed budget, reversed significance"))
        arms.append(Arm("csgqr-probe-sig-" + t, "probe_need", "significance",
                        "graded", b, "CSGQR: need-probe gate + significance"))
        arms.append(Arm("csgqr-probe-rand-" + t, "probe_need", "random_s%d" % seeds[0],
                        "graded", b, "factorial: confidence only, random order"))
        arms.append(Arm("csgqr-conf-sig-" + t, "conf_probe", "significance",
                        "graded", b, "CSGQR-v1 target: confidence probe gate"))
        arms.append(Arm("csgqr-lens-sig-" + t, "lens_maxprob", "significance",
                        "graded", b, "exact vocabulary lens gate (1 - max_prob)"))
        arms.append(Arm("csgqr-binary-probe-" + t, "probe_need", "significance",
                        "binary", b, "binary gate: whole remnant or nothing"))
        for s in seeds:
            if "probe" in perm_for:
                arms.append(Arm("randtok-graded-s%d-" % s + t,
                                "perm:csgqr-probe-sig-" + t + ":%d" % s,
                                "significance", "graded", b,
                                "CSGQR budgets permuted across tokens"))
                arms.append(Arm("randtok-binary-s%d-" % s + t,
                                "perm:csgqr-binary-probe-" + t + ":%d" % s,
                                "significance", "graded", b,
                                "binary-gate budgets permuted across tokens"))
            if "conf" in perm_for:
                arms.append(Arm("randtok-conf-s%d-" % s + t,
                                "perm:csgqr-conf-sig-" + t + ":%d" % s,
                                "significance", "graded", b,
                                "confidence-probe budgets permuted across tokens"))
            if "lens" in perm_for:
                arms.append(Arm("randtok-lens-s%d-" % s + t,
                                "perm:csgqr-lens-sig-" + t + ":%d" % s,
                                "significance", "graded", b,
                                "lens-gate budgets permuted across tokens"))
        arms.append(Arm("oracle-sig-" + t, "oracle", "significance", "graded", b,
                        "CEILING: allocate by true per-token utility"))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=sorted(SPLITS))
    ap.add_argument("--bits", default="int4", choices=sorted(SPECS))
    ap.add_argument("--budgets", default="0.10,0.20,0.30,0.50,0.75")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--only", default="", help="comma-separated arm-key substrings")
    ap.add_argument("--perm-controls-for", default="probe",
                    help="comma-separated gate families that get a "
                         "permuted-budget twin: probe, conf, lens")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--rung", type=int, default=4,
                    help="per-token budgets are multiples of this many groups. "
                         "Caps the number of distinct group sets in one forward "
                         "(and so the kernel-launch cost) without touching "
                         "matched compute: the allocation is exact on the rung "
                         "scale, so every arm still executes an identical total.")
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--significance",
                    default="experiments/audit_sgqr_csgqr/results/significance_96.json")
    ap.add_argument("--probes", default="experiments/audit_sgqr_csgqr/results/probes.pt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(20260824)

    sig = json.loads((REPO / args.significance).read_text(encoding="utf-8"))
    scores = {int(k): v for k, v in sig["scores"].items()}
    magsc = {int(k): v for k, v in sig["magnitude_scores"].items()}
    probes = torch.load(REPO / args.probes, weights_only=False)

    budgets = [float(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    arms = build_arms(budgets, seeds,
                      tuple(x for x in args.perm_controls_for.split(",") if x))
    if args.only:
        want = [w for w in args.only.split(",") if w]
        arms = [a for a in arms if any(w in a.key for w in want)]

    model, tok = load_model(device)
    prompts = load_split(tok, SPLITS[args.split], max_len=args.max_len)
    head = make_head(model)
    inst = install(model, spec=SPECS[args.bits])
    tap = ResidualTap(model, min(RETIRED))

    # ---------------------------------------------------------------- orders
    ORDERS = {"significance": global_order(scores),
              "magnitude": global_order(magsc),
              "reversed": list(reversed(global_order(scores)))}
    for s in seeds:
        ORDERS["random_s%d" % s] = random_order(20260824 + 1000 * s)
    ALLOC = {name: [alloc_from_order(order, k) for k in range(TOTAL_GROUPS + 1)]
             for name, order in ORDERS.items()}

    # ---------------------------------------------------------- pass 1: signal
    print("pass 1: dense reference, deletion damage, remnant benefit, gate signals")
    t0 = time.time()
    dense_reply: list[torch.Tensor] = []
    gold_all: list[torch.Tensor] = []
    n_pos_all: list[int] = []                 # positions per prompt (full sequence)
    reply_slice: list[tuple[int, int]] = []   # (start,end) of reply rows within prompt
    demands: dict[str, list[torch.Tensor]] = {k: [] for k in
                                              ("flat", "probe_need", "conf_probe",
                                               "lens_maxprob", "oracle")}
    per_prompt_util_reply: list[torch.Tensor] = []
    lens_reply: list[dict[str, torch.Tensor]] = []
    NONE = [{l: [] for l in RETIRED}]
    FULL = [{l: list(range(N_GROUPS)) for l in RETIRED}]
    mu, sd = probes["mu"], probes["sd"]
    w_need, w_conf = probes["probes"]["need"], probes["probes"]["confidence"]

    for pi, p in enumerate(prompts):
        ids = p["ids"].to(device)
        b = p["boundary"]
        T = ids.shape[1]
        inst.use_dense()
        with torch.no_grad():
            lg = model(ids).logits[0].float()
        dense_all = torch.log_softmax(lg, dim=-1)
        del lg
        dense_reply.append(dense_all[b - 1:-1].cpu())
        gold_all.append(ids[0, b:].cpu())
        h = tap.buf[0].detach()                      # [T, d] all positions
        pd_all = dense_all.exp()

        inst.use_remnant()
        inst.set_levels(None)
        inst.set_allocation_by_level(NONE)
        with torch.no_grad():
            del_all = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
        kl_del = (pd_all * (dense_all - del_all)).sum(dim=-1)
        del del_all
        inst.set_allocation_by_level(FULL)
        with torch.no_grad():
            rem_all = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
        kl_rem = (pd_all * (dense_all - rem_all)).sum(dim=-1)
        del rem_all, pd_all, dense_all
        inst.use_dense()

        util = (kl_del - kl_rem).cpu()               # [T] true per-position utility
        hs = ((h.float().cpu().to(torch.float64) - mu) / sd)
        ones = torch.ones(hs.shape[0], 1, dtype=torch.float64)
        need = (torch.cat([hs, ones], dim=1) @ w_need)
        conf = (torch.cat([hs, ones], dim=1) @ w_conf)
        lf = lens_features(head, h)

        demands["flat"].append(torch.ones(T, dtype=torch.float64))
        demands["probe_need"].append(need)
        demands["conf_probe"].append(-conf)          # low confidence => high demand
        demands["lens_maxprob"].append(1.0 - lf["max_prob"].to(torch.float64))
        demands["oracle"].append(util.to(torch.float64))
        n_pos_all.append(T)
        reply_slice.append((b - 1, T - 1))
        per_prompt_util_reply.append(util[b - 1:-1])
        lens_reply.append({k: v[b - 1:-1] for k, v in lf.items()})
        if (pi + 1) % 32 == 0:
            print("  %d/%d  %.0fs" % (pi + 1, len(prompts), time.time() - t0), flush=True)

    tap.close()
    total_positions = sum(n_pos_all)
    total_reply = sum(int(x.numel()) for x in per_prompt_util_reply)
    clusters = []
    for pi, x in enumerate(per_prompt_util_reply):
        clusters.extend([pi] * int(x.numel()))
    print("  %d prompts, %d total positions, %d reply positions, %.0fs"
          % (len(prompts), total_positions, total_reply, time.time() - t0))

    # Demands are pooled across prompts so the gate may move compute BETWEEN
    # prompts as well as within one. Shifting a non-negative floor on so that a
    # demand vector with negative entries (utility can be negative) still yields
    # a proportional allocation rather than a clipped one.
    pooled: dict[str, torch.Tensor] = {}
    for k, chunks in demands.items():
        v = torch.cat(chunks)
        if k != "flat":
            v = v - v.min()
        pooled[k] = v.clamp(min=0.0) + 1e-12

    # ---------------------------------------------------------- budget vectors
    RUNG = max(1, int(args.rung))
    if TOTAL_GROUPS % RUNG:
        raise SystemExit("rung %d does not divide %d groups" % (RUNG, TOTAL_GROUPS))
    MAX_RUNG = TOTAL_GROUPS // RUNG
    budget_vec: dict[str, torch.Tensor] = {}
    budget_target: dict[float, int] = {}
    for a in arms:
        if a.mode in ("dense", "none", "fixedfull"):
            continue
        R = int(round(a.budget * MAX_RUNG * total_positions))
        budget_target[a.budget] = R * RUNG
        if a.demand.startswith("perm:"):
            _, ref, sd_ = a.demand.split(":")
            if ref not in budget_vec:
                raise SystemExit(
                    "arm %s permutes the budgets of %s, which is not in this "
                    "run. A permuted control is only matched compute if its "
                    "reference arm ran in the same sweep." % (a.key, ref))
            base = budget_vec[ref]
            g = torch.Generator().manual_seed(770000 + int(sd_))
            budget_vec[a.key] = base[torch.randperm(base.numel(), generator=g)]
        elif a.mode == "binary":
            budget_vec[a.key] = binary_budgets(pooled[a.demand], R,
                                               max_k=MAX_RUNG) * RUNG
        else:
            budget_vec[a.key] = allocate_exact(pooled[a.demand], R,
                                               max_k=MAX_RUNG) * RUNG
        got = int(budget_vec[a.key].sum())
        if got != R * RUNG:
            raise SystemExit("arm %s allocated %d != %d" % (a.key, got, R * RUNG))

    # ------------------------------------------------------------ pass 2: arms
    print("\npass 2: %d arms" % len(arms))
    offsets, acc = [], 0
    for T in n_pos_all:
        offsets.append(acc)
        acc += T

    out_arms: dict[str, dict] = {}
    per_example = (RES / ("per_example_results%s.jsonl"
                          % (("_" + args.tag) if args.tag else ""))).open(
        "w", encoding="utf-8", newline="\n")
    routing_rows: list[dict] = []

    for ai, a in enumerate(arms):
        t1 = time.time()
        inst.reset_counters()
        kls, nlls, agrees, golds = [], [], [], []
        flips, first_flips = [], []
        for pi, p in enumerate(prompts):
            ids = p["ids"].to(device)
            b = p["boundary"]
            T = n_pos_all[pi]
            d_lp = dense_reply[pi].to(device)
            gold = gold_all[pi].to(device)
            if a.mode == "dense":
                inst.use_dense()
                with torch.no_grad():
                    arm_lp = torch.log_softmax(
                        model(ids).logits[0, b - 1:-1].float(), dim=-1)
            else:
                inst.use_remnant()
                if a.mode == "none":
                    inst.set_allocation_by_level(NONE)
                    inst.set_levels(None)
                elif a.mode == "fixedfull":
                    inst.set_allocation_by_level(FULL)
                    inst.set_levels(None)
                else:
                    inst.set_allocation_by_level(ALLOC[a.order])
                    k = budget_vec[a.key][offsets[pi]:offsets[pi] + T]
                    inst.set_levels(k)
                with torch.no_grad():
                    arm_lp = torch.log_softmax(
                        model(ids).logits[0, b - 1:-1].float(), dim=-1)
                inst.use_dense()
            m = position_metrics(d_lp, arm_lp, gold)
            kls.append(m["kl"].cpu())
            nlls.append(m["nll_gold"].cpu())
            agrees.append(m["agree"].cpu())
            golds.append(m["gold_top1"].cpu())
            dis = (m["agree"] < 0.5)
            flips.append(float(dis.any().item()))
            first_flips.append(float(dis.float().argmax().item()) if bool(dis.any()) else -1.0)
            del arm_lp, d_lp, m

        kl = torch.cat(kls).to(torch.float64)
        nll = torch.cat(nlls).to(torch.float64)
        agr = torch.cat(agrees).to(torch.float64)
        gld = torch.cat(golds).to(torch.float64)
        exec_groups = inst.executed_groups() if a.mode not in ("dense",) else 0
        exec_tokens = inst.executed_tokens() if a.mode not in ("dense",) else 0
        act = (exec_groups / (exec_tokens * TOTAL_GROUPS)) if exec_tokens else 0.0

        rec = {
            "arm": a.key, "label": a.label, "demand": a.demand, "order": a.order,
            "mode": a.mode, "budget": a.budget,
            "n_positions": int(kl.numel()), "n_prompts": len(prompts),
            "kl": cluster_bootstrap_mean(kl, clusters, n_boot=args.n_boot),
            "nll_gold": cluster_bootstrap_mean(nll, clusters, n_boot=args.n_boot),
            "agree": cluster_bootstrap_mean(agr, clusters, n_boot=args.n_boot),
            "gold_top1": cluster_bootstrap_mean(gld, clusters, n_boot=args.n_boot),
            "perplexity_gold": float(torch.exp(nll.mean())),
            "sequence_flip_rate": float(sum(flips) / len(flips)),
            "executed_group_tokens": int(exec_groups),
            "executed_tokens": int(exec_tokens),
            "active_fraction": act,
            "elapsed_s": round(time.time() - t1, 1),
        }
        out_arms[a.key] = rec
        routing_rows.append({k: rec[k] for k in
                             ("arm", "mode", "demand", "order", "budget",
                              "executed_group_tokens", "executed_tokens",
                              "active_fraction")})
        for i in range(int(kl.numel())):
            per_example.write(json.dumps({
                "arm": a.key, "prompt": clusters[i], "i": i,
                "kl": round(float(kl[i]), 8), "nll": round(float(nll[i]), 6),
                "agree": int(agr[i]), "gold_top1": int(gld[i])}) + "\n")
        print("  [%2d/%2d] %-26s act=%.4f  KL=%.5f  gold=%.4f  ppl=%.2f  %.0fs"
              % (ai + 1, len(arms), a.key, act, rec["kl"]["point"],
                 rec["gold_top1"]["point"], rec["perplexity_gold"],
                 rec["elapsed_s"]), flush=True)
        # keep the raw vectors for paired contrasts
        rec["_kl"] = kl
        rec["_nll"] = nll
        rec["_gold"] = gld

    per_example.close()

    # ------------------------------------------------------------- contrasts
    print("\npaired contrasts (prompt-clustered, paired at the position)")
    contrasts = []
    pvals: dict[str, float] = {}

    def pair(a_key: str, b_key: str, why: str) -> None:
        if a_key not in out_arms or b_key not in out_arms:
            return
        A, B = out_arms[a_key], out_arms[b_key]
        row = {"arm": a_key, "vs": b_key, "question": why,
               "budget": A["budget"],
               "activation_gap": A["active_fraction"] - B["active_fraction"],
               "compute_gap_group_tokens": (A["executed_group_tokens"]
                                            - B["executed_group_tokens"])}
        for metric, key in (("kl", "_kl"), ("nll_gold", "_nll"),
                            ("gold_top1", "_gold")):
            row[metric] = cluster_bootstrap_mean_pair(A[key], B[key], clusters,
                                                      n_boot=args.n_boot)
        contrasts.append(row)

    for b in budgets:
        t = format(b, ".2f")
        for s in seeds:
            pair("sgqr-sig-" + t, "sgqr-rand-s%d-" % s + t,
                 "does interventional significance beat random selection?")
        pair("sgqr-sig-" + t, "sgqr-mag-" + t,
             "does significance beat the magnitude proxy?")
        pair("sgqr-sig-" + t, "sgqr-rev-" + t,
             "does the ORDER carry the information?")
        for s in seeds:
            pair("csgqr-probe-sig-" + t, "randtok-graded-s%d-" % s + t,
                 "at EXACTLY equal compute, does the confidence gate beat "
                 "random token allocation?")
            pair("csgqr-binary-probe-" + t, "randtok-binary-s%d-" % s + t,
                 "same question for the binary gate")
        pair("csgqr-probe-sig-" + t, "sgqr-sig-" + t,
             "THE HEADLINE: does CSGQR beat SGQR at matched compute?")
        pair("csgqr-conf-sig-" + t, "sgqr-sig-" + t,
             "does the CSGQR-v1 confidence target beat SGQR?")
        pair("csgqr-lens-sig-" + t, "sgqr-sig-" + t,
             "does the exact vocabulary lens beat SGQR?")
        pair("csgqr-binary-probe-" + t, "sgqr-sig-" + t,
             "does a binary gate beat a fixed budget?")
        pair("oracle-sig-" + t, "sgqr-sig-" + t,
             "CEILING: what could a perfect token gate buy over SGQR?")
        pair("csgqr-probe-sig-" + t, "csgqr-probe-rand-" + t,
             "factorial: significance given confidence")
        pair("csgqr-probe-rand-" + t, "sgqr-rand-s%d-" % seeds[0] + t,
             "factorial: confidence given random order")

    for c in contrasts:
        lo, hi = c["kl"]["lo"], c["kl"]["hi"]
        c["kl_excludes_zero"] = bool(lo > 0 or hi < 0)

    payload = {
        "audit": "sgqr_csgqr_dual", "split": args.split,
        "split_range": list(SPLITS[args.split]),
        "prompts": len(prompts), "reply_positions": total_reply,
        "total_positions": total_positions,
        "retired_layers": list(RETIRED), "n_groups_per_layer": N_GROUPS,
        "total_groups": TOTAL_GROUPS, "bits": args.bits,
        "spec": SPECS[args.bits].name(), "budgets": budgets, "seeds": seeds,
        "significance_source": args.significance, "probes": args.probes,
        "perm_controls_for": args.perm_controls_for,
        "max_len": args.max_len, "n_boot": args.n_boot,
        "unit_of_analysis": "reply position, bootstrap clustered by prompt",
        "rung_groups": RUNG,
        "budget_target_group_tokens": {format(k, ".2f"): v
                                       for k, v in budget_target.items()},
        "matched_compute": ("every arm at a budget point executes exactly the "
                            "same number of group-token products; per-token "
                            "budgets are multiples of %d groups" % RUNG),
        "gate_signal_note": ("captured from the residual entering layer %d, "
                             "which no arm modifies, so every arm is gated on "
                             "identical values" % min(RETIRED)),
        "donor_resident": True,
        "memory_note": "donors resident for exact pairing; VRAM comes from systems.py",
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in out_arms.items()},
        "contrasts": contrasts,
        "environment": env_manifest(),
    }
    tag = ("_" + args.tag) if args.tag else ""
    (RES / ("frontier_%s%s.json" % (args.split, tag))).write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    with (RES / ("results%s.csv" % tag)).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mode", "demand", "order", "budget", "active_fraction",
                    "executed_group_tokens", "kl", "kl_lo", "kl_hi",
                    "gold_top1", "gold_lo", "gold_hi", "nll_gold",
                    "perplexity_gold", "sequence_flip_rate", "elapsed_s"])
        for k, v in out_arms.items():
            w.writerow([k, v["mode"], v["demand"], v["order"], v["budget"],
                        round(v["active_fraction"], 6), v["executed_group_tokens"],
                        v["kl"]["point"], v["kl"]["lo"], v["kl"]["hi"],
                        v["gold_top1"]["point"], v["gold_top1"]["lo"],
                        v["gold_top1"]["hi"], v["nll_gold"]["point"],
                        v["perplexity_gold"], v["sequence_flip_rate"],
                        v["elapsed_s"]])

    with (RES / ("routing_statistics%s.csv" % tag)).open("w", newline="",
                                                        encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(routing_rows[0].keys()))
        w.writeheader()
        for r in routing_rows:
            w.writerow(r)

    torch.save({"budget_vectors": budget_vec, "pooled_demands": pooled,
                "clusters": clusters, "n_pos": n_pos_all,
                "utility_reply": torch.cat(per_prompt_util_reply),
                "lens_reply": {k: torch.cat([d[k] for d in lens_reply])
                               for k in lens_reply[0]}},
               RES / ("frontier_%s%s.aux.pt" % (args.split, tag)))
    print("\nwrote frontier_%s%s.json, results%s.csv, routing_statistics%s.csv, "
          "per_example_results%s.jsonl" % (args.split, tag, tag, tag, tag))


if __name__ == "__main__":
    main()
