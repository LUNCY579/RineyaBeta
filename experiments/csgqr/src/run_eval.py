#!/usr/bin/env python3
"""Quality evaluation for CSGQR arms, protocol-identical to ``run_gates.py``.

Comparability first
-------------------
``--self-check`` runs the dense arm and asserts it reproduces the stored gate
baseline to ``1e-9``. Run it before trusting any comparison: a harness that
silently differs from the one that produced the baselines measures the harness,
not the arms. This mirrors the discipline in
``experiments/attention_shape/src/run_eval.py``, which found that a change of
bf16 GEMM kernel moved 23% of greedy replies while changing mean quality by
+0.00138 -- aggregates hide reply-level churn completely, so this script reports
both.

Protocol, taken from ``manifests/eval_assets.json`` and matched to
``scripts/eval/run_gates.py``: chat template with ``add_generation_prompt=True``
and ``enable_thinking=False``; ``max_new_tokens=32``; greedy; ``eos_only``
response control (no extra stopping criteria, no sentence trimming); reply
decoded with ``skip_special_tokens=True`` then ``finalize_response``; the visible
token count passed to the scorer as *both* ``token_count`` and
``visible_model_tokens``; corpus profile built from the dataset records.

Nothing about decoding is configurable here. That is deliberate.

Two references, and the difference matters
------------------------------------------
``b0-dense`` is the true model and is comparable to the stored baselines.
``b2-fp-all`` is the *same weights* travelling the remnant code path. It is the
correct reference for every quantized arm, because cuBLAS selects different bf16
GEMM tiles for some shapes and the remnant path is therefore not always
bit-identical to ``nn.Linear`` even with identical weights. ``|b0 - b2|`` is the
noise floor; a quality difference smaller than it is not attributable to CSGQR.

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

from envinfo import collect, require_cuda                        # noqa: E402
from quant import QuantSpec                                      # noqa: E402
from router import RoutingTrace                                  # noqa: E402
from significance import magnitude_ranking, reversed_ranking, shuffled_ranking  # noqa: E402
from surgery import install_csgqr                                # noqa: E402
from variants import CATALOGUE, Arm                              # noqa: E402

import scripts.train.memory as mem                               # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
DATASETS = {"chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
            "rag_dev": "artifacts/eval_sets/rag_dev.jsonl"}
BASELINES = {"chat_dev": "artifacts/gate_runs/r2_base_chat_dev",
             "rag_dev": "artifacts/gate_runs/r2_base_rag_dev"}
SEED = 20260823


def normalize(record: dict) -> dict:
    """Mirror of ``run_gates.py:normalize``: the gold reply leaves ``messages``."""
    messages = [dict(m) for m in record.get("messages") or []]
    reference = record.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    out = dict(record)
    out["messages"] = messages
    out["reference"] = reference
    out["category"] = str(record.get("category")
                          or (record.get("metadata") or {}).get("category")
                          or (record.get("metadata") or {}).get("reply_class")
                          or "uncategorized")
    return out


def resolve_ranking(arm: Arm, remnant, ranking_source: list[int] | None,
                    n_groups: int) -> list[int]:
    """The group ordering this arm should use.

    ``shuffled`` and ``reversed`` are derived from the *significance* ranking, so
    they are genuinely the same information with its order damaged, rather than
    an unrelated permutation.
    """
    base = list(ranking_source) if ranking_source else list(range(n_groups))
    if arm.selector == "magnitude":
        return magnitude_ranking(remnant)
    if "shuffled" in arm.key:
        return shuffled_ranking(base, seed=SEED)
    if "reversed" in arm.key:
        return reversed_ranking(base)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="b0-dense")
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tau-skip", type=float, default=None)
    ap.add_argument("--tau-full", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--feature", default=None)
    ap.add_argument("--ranking-from", default=None,
                    help="a significance JSON with a 'ranking' list")
    ap.add_argument("--calibrate-alpha", type=int, default=0,
                    help="prompts to fit the output gain on (0 = alpha stays 1)")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--compare-to", default=None,
                    help="an arm key whose results.jsonl to diff replies against")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.self_check:
        args.arm = "b0-dense"
    if args.arm not in CATALOGUE:
        raise SystemExit("unknown arm " + args.arm + "; have "
                         + ", ".join(sorted(CATALOGUE)))
    arm = CATALOGUE[args.arm]

    ds_path = REPO / DATASETS[args.dataset]
    if "SEALED" in ds_path.name:
        raise SystemExit("refusing to evaluate a sealed set")

    require_cuda()
    assets = json.loads((REPO / "manifests" / "eval_assets.json")
                        .read_text(encoding="utf-8"))
    protocol = assets["generation_protocol"]

    from scripts.common.quality_evaluation import (
        QUALITY_EVALUATION_VERSION, aggregate_quality_rows, build_corpus_profile,
        evaluate_response, load_quality_config)
    from scripts.common.response_control import (
        build_stopping_criteria, finalize_response, infer_stop_reason)

    records = [normalize(json.loads(l)) for l in
               ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]

    cap = mem.apply_cap(args.cap_mib)
    device = torch.device("cuda:0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(REPO / MODEL,
                                                 dtype=torch.bfloat16)
    model.config.use_cache = True
    model = model.to(device).eval()

    layer = args.layer
    trace = None
    install_report = None
    if not arm.dense:
        if layer is None:
            raise SystemExit("--layer is required for every arm except b0-dense")
        n_groups = 1 if arm.mode == "none" else args.n_groups
        ranking_source = None
        if args.ranking_from:
            src = json.loads((REPO / args.ranking_from).read_text(encoding="utf-8"))
            ranking_source = src.get("ranking") or src.get("significance_ranking")

        overrides = {}
        for name, val in (("tau_skip", args.tau_skip), ("tau_full", args.tau_full),
                          ("gamma", args.gamma), ("fraction", args.fraction),
                          ("feature", args.feature)):
            if val is not None:
                overrides[name] = val
        if overrides:
            arm = Arm(**{**arm.__dict__, **overrides})

        trace = RoutingTrace(n_groups)
        donor = None if arm.donor_offset is None else \
            (layer + arm.donor_offset) % model.config.num_hidden_layers
        donor_mlp = model.model.layers[donor if donor is not None else layer].mlp

        # The magnitude ranking needs the packed weights, so build the remnant
        # once with the donor still resident, read what is needed off it, then
        # install for real. alpha is fitted here too, against the *unquantized*
        # donor -- which is only possible while that donor still exists.
        ranking = list(ranking_source) if ranking_source else list(range(n_groups))
        alpha_vec = None
        if arm.selector == "magnitude" or "shuffled" in arm.key \
                or "reversed" in arm.key or args.calibrate_alpha:
            from remnant import QuantizedMlpRemnant, RemnantConfig, calibrate_alpha
            scratch = QuantizedMlpRemnant(
                donor_mlp, RemnantConfig(layer, n_groups, arm.spec,
                                         torch.bfloat16)).to(device)
            ranking = resolve_ranking(arm, scratch, ranking_source, n_groups)
            if args.calibrate_alpha and arm.spec.bits != 16:
                normed = _collect_mlp_inputs(model, tokenizer, layer, records,
                                            args.calibrate_alpha, device)
                alpha_vec = calibrate_alpha(scratch, donor_mlp, normed)
                print("alpha     fitted on " + str(normed.shape[0])
                      + " positions -> " + format(alpha_vec[0].item(), ".5f"))
            del scratch
            torch.cuda.empty_cache()

        inst = install_csgqr(model, layer, spec=arm.spec, n_groups=n_groups,
                             policy=arm.policy(ranking=ranking, n_groups=n_groups),
                             donor_layer=donor, trace=trace,
                             keep_donor_resident=False)
        if alpha_vec is not None:
            inst.module.remnant.alpha.copy_(alpha_vec.to(device))
        install_report = inst.report()

    print("arm       " + arm.key + "  -- " + arm.label)
    print("purpose   " + arm.purpose)
    print("dataset   " + args.dataset + "  (" + str(len(records)) + " cases)")
    if install_report:
        print("layer     " + str(install_report["layer"])
              + ("  donor L" + str(install_report["donor_layer"])
                 if install_report["wrong_layer_donor"] else "")
              + "   " + install_report["spec"]
              + "   " + str(install_report["n_groups"]) + " groups")
        print("packed    " + format(install_report["packed_bytes"]["total"], ",")
              + " B  (x" + format(install_report["compression_vs_dense"], ".2f")
              + " vs bf16)")
        print("policy    " + install_report["policy"]["name"])
    print("protocol  max_new_tokens=" + str(protocol["max_new_tokens"])
          + "  greedy=" + str(not protocol["do_sample"])
          + "  " + protocol["response_control"]
          + "   cap " + (format(cap, ",.0f") + " MiB" if cap else "n/a"))
    print("scorer    v" + QUALITY_EVALUATION_VERSION + "\n")

    config = load_quality_config()
    corpus_profile = build_corpus_profile(records)
    pad_id = tokenizer.pad_token_id
    rows: list[dict] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    for i, record in enumerate(records):
        rendered = tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        stopping, criterion = build_stopping_criteria(
            tokenizer, int(inputs.input_ids.shape[1]),
            str(protocol["response_control"]),
            int(protocol["min_new_tokens_before_boundary"]))
        kwargs = {"max_new_tokens": int(protocol["max_new_tokens"]),
                  "do_sample": bool(protocol["do_sample"]),
                  "pad_token_id": pad_id,
                  "eos_token_id": tokenizer.eos_token_id}
        if stopping is not None:
            kwargs["stopping_criteria"] = stopping
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        new_tokens = generated[0, inputs.input_ids.shape[1]:]

        ordinary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finalized = finalize_response(ordinary, str(protocol["response_control"]))
        reply = finalized.text
        visible = len(tokenizer(reply, add_special_tokens=False)["input_ids"]) \
            if reply else 0
        stop_reason = infer_stop_reason(
            new_tokens.tolist(), eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=int(protocol["max_new_tokens"]), criterion=criterion)
        quality = evaluate_response(record, reply, visible,
                                    visible_model_tokens=visible,
                                    config=config, corpus_profile=corpus_profile)
        rows.append({"id": str(record["id"]), "category": record["category"],
                     "reply": reply, "reference": record.get("reference"),
                     "generated_tokens": int(new_tokens.numel()),
                     "visible_generated_tokens": visible,
                     "prompt_tokens": int(inputs.input_ids.shape[1]),
                     "stop_reason": stop_reason, **quality})
        if (i + 1) % 100 == 0:
            print("  " + str(i + 1) + "/" + str(len(records)) + "  "
                  + format(time.time() - started, ".0f") + "s", flush=True)

    elapsed = time.time() - started
    summary = aggregate_quality_rows(rows)
    summary["quality_evaluation_version"] = QUALITY_EVALUATION_VERSION

    out_dir = REPO / (args.out or ("experiments/csgqr/results/eval_" + arm.key
                                   + "__" + args.dataset
                                   + ("" if arm.dense else "__L" + str(layer))))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_cat: dict[str, dict] = {}
    for r in rows:
        b = per_cat.setdefault(r["category"], {"n": 0, "total": 0.0})
        b["n"] += 1
        b["total"] += float(r.get("quality_score") or 0.0)
    (out_dir / "per_category.json").write_text(json.dumps(
        {k: {"n": v["n"], "mean_quality_score": v["total"] / v["n"]}
         for k, v in sorted(per_cat.items())}, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "arm": arm.key, "label": arm.label, "purpose": arm.purpose,
        "dataset": args.dataset,
        "dataset_sha256": assets["eval_sets"].get(
            {"chat_dev": "chat_dev_510", "rag_dev": "rag_dev_752"}[args.dataset],
            {}).get("sha256"),
        "layer": layer, "n_groups": (None if arm.dense else args.n_groups),
        "case_count": len(rows), "protocol": protocol,
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "mean_quality_score": summary["mean_quality_score"],
        "install": install_report,
        "routing": trace.summary() if trace else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1 << 20),
        "elapsed_s": round(elapsed, 1),
        "seconds_per_case": round(elapsed / max(len(rows), 1), 4),
        "environment": collect(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")

    print("\nmean_quality_score      " + format(summary["mean_quality_score"], ".5f"))
    print("quality_pass_rate       " + format(summary["quality_pass_rate"], ".5f"))
    print("intelligibility         " + format(summary["intelligibility_pass_rate"], ".5f"))
    print("critical_defect_rate    " + format(summary["critical_defect_case_rate"], ".5f"))
    print("peak allocated / reserved  "
          + format(manifest["peak_allocated_mib"], ",.1f") + " / "
          + format(manifest["peak_reserved_mib"], ",.1f") + " MiB")
    print("elapsed                 " + format(elapsed, ".1f") + "s  ("
          + format(manifest["seconds_per_case"], ".3f") + " s/case)")
    if trace:
        t = trace.summary()
        print("routing   active_fraction " + format(t["active_fraction"], ".4f")
              + "   invocation_rate " + format(t["invocation_rate"], ".4f")
              + "   skip_rate " + format(t["skip_rate"], ".4f"))
        print("          tokens gated " + str(t["tokens"])
              + "   mean groups/token " + format(t["mean_groups_per_token"], ".2f"))

    if args.self_check:
        ref = json.loads((REPO / BASELINES[args.dataset] / "summary.json")
                         .read_text(encoding="utf-8"))
        d = abs(summary["mean_quality_score"] - ref["mean_quality_score"])
        print("\nself-check vs " + BASELINES[args.dataset])
        print("  stored   " + format(ref["mean_quality_score"], ".7f"))
        print("  measured " + format(summary["mean_quality_score"], ".7f"))
        print("  |delta|  " + format(d, ".3e"))
        if args.limit:
            print("  PARTIAL run -- not a valid check; drop --limit")
        elif d < 1e-9:
            print("  PASS - this harness reproduces the stored baseline exactly")
        else:
            print("  FAIL - harness differs from the one that made the baselines")
            raise SystemExit(1)

    if args.compare_to:
        ref_path = (REPO / "experiments/csgqr/results"
                    / ("eval_" + args.compare_to + "__" + args.dataset
                       + ("" if CATALOGUE[args.compare_to].dense
                          else "__L" + str(layer))) / "results.jsonl")
        if not ref_path.exists():
            raise SystemExit("no reference run at " + str(ref_path))
        ref_rows = {json.loads(l)["id"]: json.loads(l)
                    for l in ref_path.read_text(encoding="utf-8").splitlines()
                    if l.strip()}
        shared = [r for r in rows if r["id"] in ref_rows]
        same = sum(1 for r in shared if r["reply"] == ref_rows[r["id"]]["reply"])
        ref_mean = sum(float(ref_rows[r["id"]].get("quality_score") or 0.0)
                       for r in shared) / max(len(shared), 1)
        print("\nreply diff vs " + args.compare_to + ": " + str(same) + "/"
              + str(len(shared)) + " byte-identical ("
              + format(100 * same / max(len(shared), 1), ".1f") + "%)")
        print("  quality " + format(ref_mean, ".5f") + " -> "
              + format(summary["mean_quality_score"], ".5f") + "  ("
              + format(summary["mean_quality_score"] - ref_mean, "+.5f") + ")")

    print("\nwrote " + str(out_dir.relative_to(REPO)))


@torch.no_grad()
def _collect_mlp_inputs(model, tokenizer, layer: int, records: list[dict],
                        n_prompts: int, device) -> torch.Tensor:
    """Normalized MLP inputs at ``layer``, for fitting the output gain.

    Calibration data only: the first ``n_prompts`` development prompts, prompt
    positions only (no gold reply), which keeps the gain fitted on inputs the
    deployed model would actually see at prefill and keeps it independent of the
    references the scorer compares against.
    """
    grabbed: list[torch.Tensor] = []
    handle = model.model.layers[layer].post_attention_layernorm \
        .register_forward_hook(lambda _m, _i, out: grabbed.append(out.detach()))
    try:
        for rec in records[:n_prompts]:
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            model(**tokenizer(text, return_tensors="pt").to(device))
    finally:
        handle.remove()
    return torch.cat([g.reshape(-1, g.shape[-1]) for g in grabbed], dim=0)


if __name__ == "__main__":
    main()
