#!/usr/bin/env python3
"""Teacher-forced, paired, properly powered evaluation of CSGQR arms.

Why this script exists
----------------------
``run_eval.py`` is the repository's generation-and-score protocol and it is the
right instrument for "would a user notice". It is the *wrong* instrument for
"does this routing policy work", and the Tier-1 sweep proved it rather than
assuming it. Measured on ``chat_dev`` (510 cases, layer 20):

* deleting the MLP changes **94.5%** of greedy replies, yet
* **55.5%** of cases score *exactly* the same before and after, and
* the paired standard deviation of the per-case quality delta is **0.149**
  against a deletion effect of **0.0114** -- a factor of 13.

So the sample size needed for 80% power is ~1,350 cases for deletion damage
itself and ~10,900 for the arm-to-arm differences of ~0.004 that the routing
question turns on. ``chat_dev`` has 510. After Holm-Bonferroni across the nine
planned contrasts, **not one was significant** -- including deletion damage. The
scorer is a coarse saturated rubric (median 0.720, p10 0.720); it cannot resolve
one MLP layer on a 0.47B model at this n, and no amount of GPU time on that
endpoint changes the arithmetic.

This script measures the same arms on the same prompts with an endpoint that
*can* resolve them, by changing the unit of analysis from the case to the token
while keeping the inference honest about clustering.

What is measured, per reply position
------------------------------------
The gold assistant turn is teacher-forced, and only positions that predict a
reply token are scored -- a divergence averaged over prompt positions is
dominated by tokens no arm ever has to produce.

``kl``          ``KL(p_dense || p_arm)`` in nats. Self-consistency with the
                unmodified model. Zero by construction for the dense arm, which
                is the harness's own null check.
``nll_arm``     ``-log p_arm(gold)``. This is a **quality** measure, not a
                self-consistency one: it asks whether the arm predicts the
                reference reply, so an arm cannot score well by confidently
                agreeing with a dense model that is itself wrong.
``d_nll``       ``nll_arm - nll_dense``, the paired quantity. Lower is better.
``agree``       ``1[argmax_arm == argmax_dense]``. The quality scorer only ever
                sees the argmax, so KL and agreement can move independently and
                both are reported.
``gold_top1``   ``1[argmax_arm == gold]``. Absolute, not relative to dense.

Sequence level, because token agreement is not sequence quality
---------------------------------------------------------------
Local errors accumulate: one flipped argmax diverges a whole greedy reply. So
each prompt also gets ``any_flip`` (did any reply position's argmax differ from
dense) and ``first_flip`` (where), and these are reported as prompt-level rates
alongside the token-level means. A method can look excellent per token and
change most replies; that gap is the point of reporting both.

Pairing and inference
---------------------
Every arm sees the same prompts, the same positions and the same gold tokens, so
all contrasts are paired at the position level. Positions within a prompt share a
context and a truncated prefix and are **not** independent, so every interval is
a **prompt-clustered** bootstrap (``stats.cluster_bootstrap``), resampling whole
prompts. Treating positions as independent would shrink every interval by roughly
the square root of the mean reply length and is the standard way to manufacture
significance here.

Exact pairing without re-quantization
-------------------------------------
For each prompt the script runs two forwards -- one dense, one through the arm --
by swapping ``layer.mlp`` between the original module and the installed CSGQR
module. That is a pointer swap, so the remnant is quantized once per arm rather
than once per prompt, and every group is measured against exactly the same packed
weights. It also means the dense reference is recomputed in the same process,
same session and same allocator state as the arm, which is what makes the pairing
exact.

The donor is therefore kept **resident** here (``keep_donor_resident=True``).
That is correct for a quality measurement and wrong for a memory one: this script
must not be used to report VRAM. ``bench_gpu.py`` frees the donor and is the only
source for a memory number.

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from envinfo import collect, require_cuda                          # noqa: E402
from measure_layers import DATASETS, load_prompts                  # noqa: E402
from router import RoutingTrace                                    # noqa: E402
from significance import (magnitude_ranking, reversed_ranking,     # noqa: E402
                          shuffled_ranking)
from stats import (cluster_bootstrap_mean,                         # noqa: E402
                   cluster_bootstrap_mean_pair)
from surgery import install_csgqr                                  # noqa: E402
from variants import CATALOGUE                                     # noqa: E402

import scripts.train.memory as mem                                 # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
SEED = 20260823


@torch.no_grad()
def reply_logprobs(model, ids: torch.Tensor, boundary: int) -> torch.Tensor:
    """Log-probs over the positions that predict the reply tokens, float32.

    Position ``boundary-1`` predicts the first reply token, so the slice starts
    there and stops one short of the end. Computed in float32 because a bf16
    log-softmax over 32768 classes carries enough error to move these
    divergences in the third decimal, which is the order of the differences
    being resolved.
    """
    return torch.log_softmax(
        model(ids).logits[0, boundary - 1:-1].float(), dim=-1)


def position_metrics(dense_lp: torch.Tensor, arm_lp: torch.Tensor,
                     gold: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-position metrics from two log-prob matrices and the gold tokens."""
    p = dense_lp.exp()
    kl = (p * (dense_lp - arm_lp)).sum(dim=-1)
    d_top1 = dense_lp.argmax(dim=-1)
    a_top1 = arm_lp.argmax(dim=-1)
    idx = gold.reshape(-1, 1)
    return {
        "kl": kl,
        "nll_arm": -arm_lp.gather(1, idx).squeeze(1),
        "nll_dense": -dense_lp.gather(1, idx).squeeze(1),
        "agree": (a_top1 == d_top1).to(torch.float32),
        "gold_top1": (a_top1 == gold).to(torch.float32),
        "gold_top1_dense": (d_top1 == gold).to(torch.float32),
    }


def build_arm(model, arm, layer: int, n_groups: int, ranking_source, device,
              utility_fn=None, gate_fn=None):
    """Install one arm, returning ``(installed, original_mlp, trace)``.

    ``original_mlp`` is handed back so the caller can swap it in for the dense
    reference forward. The donor stays resident precisely so that swap is
    possible.
    """
    n = 1 if arm.mode == "none" else n_groups
    src = list(ranking_source) if ranking_source and len(ranking_source) == n \
        else list(range(n))

    ranking = src
    if arm.selector == "magnitude" or "shuffled" in arm.key or "reversed" in arm.key:
        from remnant import QuantizedMlpRemnant, RemnantConfig     # noqa: PLC0415
        scratch = QuantizedMlpRemnant(
            model.model.layers[layer].mlp,
            RemnantConfig(layer, n, arm.spec, torch.bfloat16)).to(device)
        if arm.selector == "magnitude":
            ranking = magnitude_ranking(scratch)
        elif "shuffled" in arm.key:
            ranking = shuffled_ranking(src, seed=SEED)
        elif "reversed" in arm.key:
            ranking = reversed_ranking(src)
        del scratch
        torch.cuda.empty_cache()

    donor = None if arm.donor_offset is None else \
        (layer + arm.donor_offset) % model.config.num_hidden_layers
    trace = RoutingTrace(n)
    inst = install_csgqr(model, layer, spec=arm.spec, n_groups=n,
                         policy=arm.policy(ranking=ranking, n_groups=n),
                         donor_layer=donor, trace=trace,
                         utility_fn=utility_fn, gate_fn=gate_fn,
                         keep_donor_resident=True)
    return inst, inst.original_mlp, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="b1-delete,b4-int4-all",
                    help="comma-separated arm keys from variants.CATALOGUE")
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--prompts", type=int, default=510)
    ap.add_argument("--skip-prompts", type=int, default=0,
                    help="discard this many usable prompts first. Required when "
                         "evaluating probe-gated arms: the probes were fitted on "
                         "a split of the trace's prompts, so an overlapping "
                         "evaluation would score them on their own training data")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--ranking-from", default=None)
    ap.add_argument("--router", default=None,
                    help="a trained group-utility router (train_router.py)")
    ap.add_argument("--probe", default=None,
                    help="a trained need probe for probe_need-gated arms")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--tau-skip", type=float, default=None)
    ap.add_argument("--tau-full", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--feature", default=None)
    ap.add_argument("--tag", default=None, help="suffix for the output filename")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    keys = [k.strip() for k in args.arms.split(",") if k.strip()]
    unknown = [k for k in keys if k not in CATALOGUE]
    if unknown:
        raise SystemExit("unknown arm(s) " + ", ".join(unknown)
                         + "; have " + ", ".join(sorted(CATALOGUE)))
    ds_path = REPO / DATASETS[args.dataset]
    if "SEALED" in ds_path.name:
        raise SystemExit("refusing to evaluate a sealed set")

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False

    # Probes are loaded before the prompts so the contamination check below can
    # run before any GPU time is spent.
    ranking_source = None
    if args.ranking_from:
        src = json.loads((REPO / args.ranking_from).read_text(encoding="utf-8"))
        ranking_source = src.get("ranking") or src.get("significance_ranking")

    utility_fn = gate_fn = None
    router_meta = probe_meta = None
    if args.router:
        from probe_runtime import load_group_router                 # noqa: PLC0415
        utility_fn, router_meta = load_group_router(REPO / args.router, device)
    if args.probe:
        from probe_runtime import load_need_probe                   # noqa: PLC0415
        gate_fn, probe_meta = load_need_probe(REPO / args.probe, device)

    prompts = load_prompts(ds_path, tokenizer, args.prompts, args.max_len,
                           skip=args.skip_prompts)
    if not prompts:
        raise SystemExit("no usable prompts loaded")

    # Refuse a probe-gated or router-gated run that overlaps the probe's own
    # training prompts. The probe metadata records how many prompts its trace
    # used; anything at or below that index is contaminated.
    for meta, what in ((router_meta, "router"), (probe_meta, "need probe")):
        if meta is None:
            continue
        used = int(meta.get("train_prompts", 0)) + int(meta.get("val_prompts", 0))
        trace_prompts = used + int(meta.get("test_prompts", 0))
        if args.skip_prompts < trace_prompts:
            raise SystemExit(
                "the " + what + " was fitted from a trace over the first "
                + str(trace_prompts) + " usable prompts of "
                + str(meta.get("dataset")) + ", but this run starts at prompt "
                + str(args.skip_prompts) + ". Pass --skip-prompts "
                + str(trace_prompts) + " or more, or the arm is scored partly "
                "on its own training data.")

    n_pos = sum(p["reply_tokens"] for p in prompts)
    print("dataset  " + args.dataset + "   " + str(len(prompts)) + " prompts"
          + (" (skipping the first " + str(args.skip_prompts) + ")"
             if args.skip_prompts else "")
          + ", " + str(n_pos) + " reply positions   layer " + str(args.layer))
    print("arms     " + ", ".join(keys))
    print("metric   paired per-position KL / NLL / argmax, prompt-clustered "
          "bootstrap (" + str(args.n_boot) + " draws)\n")

    # -- dense reference, computed once and held on CPU in float16 ----------
    # float16 halves the residency and costs nothing: these are only ever used
    # to form a KL against a float32 arm distribution, and they are re-widened
    # to float32 before any arithmetic.
    started = time.time()
    dense_ref: list[torch.Tensor] = []
    for p in prompts:
        lp = reply_logprobs(model, p["ids"].to(device), p["boundary"])
        dense_ref.append(lp.to(torch.float16).cpu())
    held = sum(t.numel() * t.element_size() for t in dense_ref) / (1 << 20)
    print("dense reference cached in " + format(time.time() - started, ".1f")
          + "s   (" + format(held, ",.0f") + " MiB on CPU)")

    overrides = {k: v for k, v in (("tau_skip", args.tau_skip),
                                   ("tau_full", args.tau_full),
                                   ("gamma", args.gamma),
                                   ("fraction", args.fraction),
                                   ("feature", args.feature)) if v is not None}

    dest = REPO / (args.out or
                   ("experiments/csgqr/results/forced_"
                    + (args.tag or ("L" + str(args.layer) + "_" + args.dataset))
                    + ".json"))
    dest.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "endpoint": "teacher_forced_paired",
        "dataset": args.dataset, "layer": args.layer,
        "n_groups": args.n_groups,
        "prompts": len(prompts), "reply_positions": n_pos,
        "skip_prompts": args.skip_prompts, "max_len": args.max_len,
        "unit_of_analysis": "reply position, bootstrap clustered by prompt",
        "n_boot": args.n_boot,
        "ranking_from": args.ranking_from,
        "router": router_meta, "probe": probe_meta,
        "overrides": overrides,
        "baseline_for_contrasts": keys[0],
        "donor_resident": True,
        "memory_note": ("the donor is kept resident so the dense reference can "
                        "be recomputed by a pointer swap; VRAM numbers must come "
                        "from bench_gpu.py, not from this run"),
        "environment": collect(),
    }

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    def checkpoint() -> None:
        """Write what has completed so far.

        Called after every arm. A sweep that dies on arm 12 of 16 must not
        discard the eleven arms that already ran -- the first version of this
        script wrote once at the end and a device-mismatch bug in the router
        threw away the whole run's GPU time.
        """
        clean = {k: {kk: vv for kk, vv in v.items()
                     if not kk.startswith("_")} for k, v in results.items()}
        dest.write_text(json.dumps(
            {**header, "arms": clean, "contrasts": [], "failures": failures,
             "complete": False}, indent=2) + "\n", encoding="utf-8")

    def run_arm(key: str) -> None:
        """Evaluate one arm and record it. Raises on failure."""
        arm = CATALOGUE[key]
        if overrides:
            from variants import Arm                               # noqa: PLC0415
            arm = Arm(**{**arm.__dict__, **overrides})
        t0 = time.time()

        if arm.dense:
            inst = original = trace = None
        else:
            inst, original, trace = build_arm(
                model, arm, args.layer, args.n_groups, ranking_source, device,
                utility_fn=utility_fn, gate_fn=gate_fn)

        cols: dict[str, list[torch.Tensor]] = {}
        clusters: list[int] = []
        seq_flip: list[float] = []
        first_flip: list[float] = []
        target = model.model.layers[args.layer]

        for pi, p in enumerate(prompts):
            ids = p["ids"].to(device)
            b = p["boundary"]
            gold = ids[0, b:]

            if inst is None:
                # The dense arm is its own reference; KL is identically zero and
                # this is the harness's null check rather than a wasted pass.
                arm_lp = dense_ref[pi].to(device).float()
            else:
                target.mlp = inst.module
                arm_lp = reply_logprobs(model, ids, b)
                target.mlp = original

            d_lp = dense_ref[pi].to(device).float()
            m = position_metrics(d_lp, arm_lp, gold)
            for k, v in m.items():
                cols.setdefault(k, []).append(v.cpu())
            clusters.extend([pi] * int(gold.numel()))
            flips = (m["agree"] < 0.5)
            seq_flip.append(float(bool(flips.any())))
            first_flip.append(float(int(flips.nonzero()[0].item())
                                    if bool(flips.any()) else -1))

            if (pi + 1) % 128 == 0:
                print("    " + key + "  " + str(pi + 1) + "/" + str(len(prompts))
                      + "  " + format(time.time() - t0, ".0f") + "s", flush=True)

        if inst is not None:
            target.mlp = inst.module
            inst.hook_handle.remove()
            target.mlp = original

        flat = {k: torch.cat(v).to(torch.float64) for k, v in cols.items()}
        flat["d_nll"] = flat["nll_arm"] - flat["nll_dense"]
        cl = clusters

        def boot(name: str) -> dict:
            return cluster_bootstrap_mean(flat[name], cl, n_boot=args.n_boot)

        entry = {
            "arm": key, "label": arm.label, "purpose": arm.purpose,
            "spec": arm.spec.name(), "mode": arm.mode,
            "selector": arm.selector, "feature": arm.feature,
            "dense": bool(arm.dense),
            "n_positions": int(flat["kl"].numel()),
            "n_prompts": len(prompts),
            "kl": boot("kl"),
            "nll_arm": boot("nll_arm"),
            "d_nll": boot("d_nll"),
            "agree": boot("agree"),
            "gold_top1": boot("gold_top1"),
            "sequence_flip_rate": float(sum(seq_flip) / len(seq_flip)),
            "mean_first_flip_position": (
                float(sum(x for x in first_flip if x >= 0)
                      / max(sum(1 for x in first_flip if x >= 0), 1))),
            "install": inst.report() if inst is not None else None,
            "routing": trace.summary() if trace is not None else None,
            "elapsed_s": round(time.time() - t0, 1),
        }
        results[key] = entry
        print("  {:<18s} KL {:>8.5f} [{:>7.5f},{:>7.5f}]  dNLL {:>+8.5f}"
              "  agree {:>6.4f}  goldtop1 {:>6.4f}  seqflip {:>6.4f}".format(
                  key, entry["kl"]["point"], entry["kl"]["lo"],
                  entry["kl"]["hi"], entry["d_nll"]["point"],
                  entry["agree"]["point"], entry["gold_top1"]["point"],
                  entry["sequence_flip_rate"]))
        results[key]["_flat"] = flat        # kept in-process for the contrasts
        results[key]["_clusters"] = cl

    for key in keys:
        try:
            run_arm(key)
        except Exception as exc:                       # noqa: BLE001
            # One arm's failure must not discard the sweep. It is
            # recorded in the output so a missing arm is visibly a
            # failure rather than silently absent, and the layer's mlp
            # is restored so the next arm starts from a clean model.
            failures[key] = repr(exc)
            print("  [FAIL] " + key + ": " + repr(exc), flush=True)
            results.pop(key, None)
        checkpoint()

    # -- paired contrasts between every arm and the first one --------------
    base_key = keys[0]
    contrasts = []
    if len(keys) > 1 and base_key in results:
        print("\nPaired contrasts against " + base_key
              + " (same positions, prompt-clustered)")
        for key in keys[1:]:
            if key not in results:
                continue
            row = {"arm": key, "baseline": base_key}
            for metric in ("kl", "d_nll", "agree", "gold_top1"):
                ci = cluster_bootstrap_mean_pair(
                    results[key]["_flat"][metric],
                    results[base_key]["_flat"][metric],
                    results[key]["_clusters"], n_boot=args.n_boot)
                row[metric] = ci
            contrasts.append(row)
            print("  {:<18s} dKL {:>+9.5f} [{:>+8.5f},{:>+8.5f}] {:>4s}"
                  "   dGoldTop1 {:>+8.5f} {:>4s}".format(
                      key, row["kl"]["point"], row["kl"]["lo"], row["kl"]["hi"],
                      "SIG" if row["kl"]["excludes_zero"] else "ns",
                      row["gold_top1"]["point"],
                      "SIG" if row["gold_top1"]["excludes_zero"] else "ns"))

    for v in results.values():
        v.pop("_flat", None)
        v.pop("_clusters", None)

    out = {**header, "arms": results, "contrasts": contrasts,
           "failures": failures, "complete": not failures}
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("\n" + str(len(failures)) + " arm(s) FAILED: "
              + ", ".join(sorted(failures)))
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
