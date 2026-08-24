#!/usr/bin/env python3
"""SGQR-v2 arm runner: four retired layers, paired teacher-forced evaluation.

Endpoint choice, and why it is not the generation scorer
--------------------------------------------------------
CSGQR-v1 measured that the repository's generation-and-score endpoint cannot
resolve single-layer effects: paired sd 0.14901 against a deletion effect of
0.01135, needing ~1,350 cases where 510 exist, and after Holm-Bonferroni *no*
contrast was significant -- including deletion damage itself. Four-layer deletion
should be far more damaging, so that endpoint may become usable here. Whether it
does is an empirical question, answered by `--also-generation`, not assumed.

The primary endpoint is paired at the reply position with prompt-clustered
intervals, and it reports **two different things** that CSGQR-v1 showed can move
independently:

``kl``          fidelity to the dense model.
``gold_top1``   whether the arm predicts the *reference* reply token. This is the
                capability metric. At one layer, deletion moved KL by 0.28 nats
                and moved gold_top1 by exactly 0.00000 -- so a mechanism that
                repairs logits but not capability must not be called a recovery,
                and reporting only KL would have hidden that.
``nll_gold``    cross-entropy on the gold token; perplexity is exp() of its mean.

Exact pairing without re-quantization
-------------------------------------
Each prompt is run twice -- once dense, once through the arm -- by swapping
``layer.mlp`` between the original module and the installed remnant. That is a
pointer swap, so the four remnants are quantized once per arm rather than once
per prompt, and the dense reference is recomputed in the same process and
allocator state as the arm. The donors therefore stay resident, which is correct
for a quality measurement and **wrong for a memory one**: VRAM numbers must come
from the profiling script, and the output JSON says so.

Sealed sets are refused outright.
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
from stats import (cluster_bootstrap_mean,                         # noqa: E402
                   cluster_bootstrap_mean_pair)

from multilayer import (RetirementPlan, allocate_global,           # noqa: E402
                        allocate_random, allocate_uniform,
                        install_sgqr, magnitude_scores)

import scripts.train.memory as mem                                 # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
SEED = 20260824

#: Preregistered retirement set: the four lowest-damage layers in the committed
#: 28-layer scan (experiments/csgqr/results/layer_scan.json). Fixed before any
#: four-layer number existed. See PREREGISTRATION.md section 3.
RETIRED = (13, 14, 24, 25)

SPECS = {"fp16": QuantSpec(16, 0, True),
         "int8": QuantSpec(8, 128, False),
         "int4": QuantSpec(4, 128, False)}


@torch.no_grad()
def reply_logprobs(model, ids: torch.Tensor, boundary: int) -> torch.Tensor:
    """Log-probs over the positions that predict the reply tokens, float32."""
    return torch.log_softmax(
        model(ids).logits[0, boundary - 1:-1].float(), dim=-1)


def position_metrics(dense_lp, arm_lp, gold):
    p = dense_lp.exp()
    d_top1 = dense_lp.argmax(dim=-1)
    a_top1 = arm_lp.argmax(dim=-1)
    idx = gold.reshape(-1, 1)
    return {
        "kl": (p * (dense_lp - arm_lp)).sum(dim=-1),
        "nll_gold": -arm_lp.gather(1, idx).squeeze(1),
        "nll_gold_dense": -dense_lp.gather(1, idx).squeeze(1),
        "agree": (a_top1 == d_top1).to(torch.float32),
        "gold_top1": (a_top1 == gold).to(torch.float32),
    }


# --------------------------------------------------------------------------
def build_arms(args, sig_scores, n_groups):
    """The preregistered arm matrix, as data.

    Each entry is (key, label, spec, allocation-builder). Keeping them in one
    table means an arm-to-arm difference is a difference of configuration and
    never of implementation -- the property that let CSGQR-v1's S1 hide, where a
    fast path silently swapped a dynamic arm for a static one.
    """
    L = RETIRED
    NG = n_groups
    arms: list[tuple] = []

    arms.append(("A-dense", "A dense reference", None, None))
    arms.append(("B-delete4", "B four-layer deletion", SPECS["int4"],
                 lambda: {l: [] for l in L}))

    for bits in ("fp16", "int8", "int4"):
        arms.append(("C-" + bits + "-full",
                     "C four exact " + bits + " remnants, rho=1", SPECS[bits],
                     lambda: {l: list(range(NG)) for l in L}))

    for frac in args.budgets:
        tag = format(frac, ".2f")
        if sig_scores:
            arms.append(("D-sig-" + tag,
                         "D SGQR-v2 significance global top-k, rho=" + tag,
                         SPECS["int4"],
                         lambda f=frac: allocate_global(sig_scores, f, n_groups=NG)))
            arms.append(("D-siguni-" + tag,
                         "D' significance, per-layer uniform, rho=" + tag,
                         SPECS["int4"],
                         lambda f=frac: allocate_uniform(sig_scores, f, n_groups=NG)))
        for seed in args.random_seeds:
            arms.append(("E-rand" + str(seed) + "-" + tag,
                         "E random groups seed " + str(seed) + ", rho=" + tag,
                         SPECS["int4"],
                         lambda f=frac, s=seed: allocate_random(L, f, n_groups=NG,
                                                                seed=s)))
        arms.append(("F-mag-" + tag, "F magnitude groups, rho=" + tag,
                     SPECS["int4"], ("magnitude", frac)))

    # G: wrong-layer donors at the identical byte budget. Several mappings, not
    # one arbitrary example -- a single mapping cannot distinguish "the wrong
    # layer" from "that particular wrong layer".
    for name, shift in (("G-shift1", 1), ("G-shift6", 6), ("G-rev", None)):
        arms.append((name + "-full", name + " wrong-layer donors, rho=1",
                     SPECS["int4"], ("wrong", shift)))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all",
                    help="'all' or a comma-separated list of arm keys")
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--prompts", type=int, default=512)
    ap.add_argument("--skip-prompts", type=int, default=96,
                    help="held-out offset; 96 keeps evaluation disjoint from the "
                         "significance-calibration prompts")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--budgets", default="0.10,0.25,0.50,0.75")
    ap.add_argument("--random-seeds", default="1,2,3")
    ap.add_argument("--significance", default=None,
                    help="a JSON of per-layer group scores from calibrate.py")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--out", default="experiments/sgqr_v2/results/arms.json")
    args = ap.parse_args()
    args.budgets = [float(x) for x in args.budgets.split(",") if x]
    args.random_seeds = [int(x) for x in args.random_seeds.split(",") if x]

    ds_path = REPO / DATASETS[args.dataset]
    if "SEALED" in ds_path.name:
        raise SystemExit("refusing to evaluate a sealed set")

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(SEED)

    sig_scores = None
    sig_meta = None
    if args.significance:
        blob = json.loads((REPO / args.significance).read_text(encoding="utf-8"))
        sig_scores = {int(k): v for k, v in blob["scores"].items()}
        sig_meta = {k: v for k, v in blob.items() if k != "scores"}
        if tuple(sorted(sig_scores)) != tuple(sorted(RETIRED)):
            raise SystemExit("significance file covers layers "
                             + str(sorted(sig_scores)) + ", expected "
                             + str(sorted(RETIRED)))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False

    prompts = load_prompts(ds_path, tokenizer, args.prompts, args.max_len,
                           skip=args.skip_prompts)
    if not prompts:
        raise SystemExit("no usable prompts loaded")
    n_pos = sum(p["reply_tokens"] for p in prompts)

    all_arms = build_arms(args, sig_scores, args.n_groups)
    wanted = ([k for k, *_ in all_arms] if args.arms == "all"
              else [a.strip() for a in args.arms.split(",") if a.strip()])
    known = {k for k, *_ in all_arms}
    unknown = [k for k in wanted if k not in known]
    if unknown:
        raise SystemExit("unknown arm(s): " + ", ".join(unknown)
                         + "\nhave: " + ", ".join(sorted(known)))

    print("retired layers  " + str(list(RETIRED))
          + "   (preregistered: four lowest-damage in layer_scan.json)")
    print("dataset  " + args.dataset + "   " + str(len(prompts)) + " prompts"
          + " (skipping first " + str(args.skip_prompts) + "), "
          + str(n_pos) + " reply positions")
    print("arms     " + str(len(wanted)) + "   budgets " + str(args.budgets))
    print("metric   paired per-position KL / gold-token accuracy / NLL, "
          "prompt-clustered bootstrap\n")

    # -- dense reference, computed once -----------------------------------
    t0 = time.time()
    dense_ref = []
    for p in prompts:
        lp = reply_logprobs(model, p["ids"].to(device), p["boundary"])
        dense_ref.append(lp.to(torch.float16).cpu())
    print("dense reference cached in " + format(time.time() - t0, ".1f") + "s   ("
          + format(sum(t.numel() * t.element_size() for t in dense_ref) / (1 << 20),
                   ",.0f") + " MiB on CPU)")

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "campaign": "sgqr_v2", "endpoint": "teacher_forced_paired",
        "retired_layers": list(RETIRED),
        "layer_selection_rule": ("four lowest deletion damage in "
                                 "experiments/csgqr/results/layer_scan.json, "
                                 "fixed before any four-layer measurement"),
        "dataset": args.dataset, "prompts": len(prompts),
        "reply_positions": n_pos, "skip_prompts": args.skip_prompts,
        "max_len": args.max_len, "n_groups_per_layer": args.n_groups,
        "budgets": args.budgets, "random_seeds": args.random_seeds,
        "n_boot": args.n_boot, "significance": sig_meta,
        "unit_of_analysis": "reply position, bootstrap clustered by prompt",
        "donor_resident": True,
        "memory_note": ("donors stay resident so the dense reference is a pointer "
                        "swap; VRAM must come from the profiling script"),
        "environment": collect(),
    }
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    def checkpoint():
        clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in results.items()}
        dest.write_text(json.dumps({**header, "arms": clean,
                                    "failures": failures, "complete": False},
                                   indent=2) + "\n", encoding="utf-8")

    pristine = {l: model.model.layers[l].mlp for l in RETIRED}

    def run_arm(key, label, spec, alloc_spec):
        t = time.time()
        inst = None
        if spec is not None:
            plan = RetirementPlan(layers=RETIRED, spec=spec,
                                  n_groups=args.n_groups)
            if isinstance(alloc_spec, tuple) and alloc_spec[0] == "wrong":
                shift = alloc_spec[1]
                n_layers = model.config.num_hidden_layers
                if shift is None:            # reverse the mapping among retired
                    donors = dict(zip(RETIRED, tuple(reversed(RETIRED))))
                else:
                    donors = {l: (l + shift) % n_layers for l in RETIRED}
                plan = RetirementPlan(layers=RETIRED, spec=spec,
                                      n_groups=args.n_groups, donor_map=donors)
                alloc = {l: list(range(args.n_groups)) for l in RETIRED}
            elif isinstance(alloc_spec, tuple) and alloc_spec[0] == "magnitude":
                alloc = None                 # filled after install; needs weights
            else:
                alloc = alloc_spec()
            inst = install_sgqr(model, plan, allocation=alloc,
                                keep_donor_resident=True)
            if alloc is None:                # magnitude control
                inst.set_allocation(allocate_global(
                    magnitude_scores(inst), alloc_spec[1], n_groups=args.n_groups))

        cols: dict[str, list] = {}
        clusters: list[int] = []
        for pi, p in enumerate(prompts):
            ids = p["ids"].to(device)
            b = p["boundary"]
            gold = ids[0, b:]
            d_lp = dense_ref[pi].to(device).float()
            if inst is None:
                a_lp = d_lp
            else:
                for l in RETIRED:
                    model.model.layers[l].mlp = inst.modules[l]
                a_lp = reply_logprobs(model, ids, b)
                for l in RETIRED:
                    model.model.layers[l].mlp = pristine[l]
            for k, v in position_metrics(d_lp, a_lp, gold).items():
                cols.setdefault(k, []).append(v.cpu())
            clusters.extend([pi] * int(gold.numel()))

        flat = {k: torch.cat(v).to(torch.float64) for k, v in cols.items()}
        flat["d_nll_gold"] = flat["nll_gold"] - flat["nll_gold_dense"]
        boot = lambda n: cluster_bootstrap_mean(flat[n], clusters,
                                                n_boot=args.n_boot)
        entry = {
            "arm": key, "label": label,
            "spec": None if spec is None else spec.name(),
            "n_positions": int(flat["kl"].numel()), "n_prompts": len(prompts),
            "kl": boot("kl"), "gold_top1": boot("gold_top1"),
            "nll_gold": boot("nll_gold"), "d_nll_gold": boot("d_nll_gold"),
            "agree": boot("agree"),
            "perplexity_gold": float(torch.exp(flat["nll_gold"].mean()).item()),
            "install": inst.report() if inst is not None else None,
            "elapsed_s": round(time.time() - t, 1),
        }
        results[key] = entry
        results[key]["_flat"] = flat
        results[key]["_clusters"] = clusters
        r = entry["install"] or {}
        print("  {:<20s} KL {:>8.5f} [{:>7.5f},{:>7.5f}]  gold {:>6.4f}"
              "  ppl {:>8.3f}  rho {:>5s}  {:>6.0f}s".format(
                  key, entry["kl"]["point"], entry["kl"]["lo"], entry["kl"]["hi"],
                  entry["gold_top1"]["point"], entry["perplexity_gold"],
                  format(r.get("active_fraction", 0.0), ".3f") if r else "  -",
                  entry["elapsed_s"]), flush=True)

    for key, label, spec, alloc in all_arms:
        if key not in wanted:
            continue
        try:
            run_arm(key, label, spec, alloc)
        except Exception as exc:                                   # noqa: BLE001
            failures[key] = repr(exc)
            print("  [FAIL] " + key + ": " + repr(exc), flush=True)
            results.pop(key, None)
        finally:
            for l in RETIRED:
                model.model.layers[l].mlp = pristine[l]
        checkpoint()

    # -- paired contrasts against four-layer deletion ----------------------
    base = "B-delete4"
    contrasts = []
    if base in results:
        print("\nPaired contrasts against " + base)
        for key in results:
            if key == base:
                continue
            row = {"arm": key, "baseline": base}
            for m in ("kl", "gold_top1", "nll_gold"):
                row[m] = cluster_bootstrap_mean_pair(
                    results[key]["_flat"][m], results[base]["_flat"][m],
                    results[key]["_clusters"], n_boot=args.n_boot)
            contrasts.append(row)
            print("  {:<20s} dKL {:>+9.5f} {:<4s}  dGold {:>+8.5f} {:<4s}"
                  "  dNLL {:>+8.5f} {:<4s}".format(
                      key, row["kl"]["point"],
                      "SIG" if row["kl"]["excludes_zero"] else "ns",
                      row["gold_top1"]["point"],
                      "SIG" if row["gold_top1"]["excludes_zero"] else "ns",
                      row["nll_gold"]["point"],
                      "SIG" if row["nll_gold"]["excludes_zero"] else "ns"))

    for v in results.values():
        v.pop("_flat", None)
        v.pop("_clusters", None)
    dest.write_text(json.dumps({**header, "arms": results,
                                "contrasts": contrasts,
                                "baseline_for_contrasts": base,
                                "failures": failures,
                                "complete": not failures}, indent=2) + "\n",
                    encoding="utf-8")
    if failures:
        print("\n" + str(len(failures)) + " arm(s) FAILED: "
              + ", ".join(sorted(failures)))
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
