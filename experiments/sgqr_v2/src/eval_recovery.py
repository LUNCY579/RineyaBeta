#!/usr/bin/env python3
"""Evaluate a trained recovery arm on the identical endpoint as the remnant arms.

Arms H (same-byte LoRA) and I (same-byte generic adapter) have to be measured
against exactly the same dense reference, on exactly the same held-out prompts, at
exactly the same reply positions as arms A-G. Anything else and the capacity
comparison is confounded by the harness.

So this script recomputes the dense reference itself, in the same process, from
the pristine checkpoint, *before* any surgery -- and then emits the same JSON
schema as ``run_arms.py`` so the two files merge without translation.

What "the same bytes" means here
--------------------------------
Reported for every arm: trainable parameter count, stored bytes at the shipping
dtype (bf16), and bytes actually on disk. The remnant's budget is measured, not
estimated -- 4 x 4,939,872 B, a figure where the analytic model and
``memory_allocated`` agree exactly. Matching parameter *count* would be the wrong
comparison, because packed uint8 + fp16 scales and dense bf16 do not cost the
same per parameter.
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
from stats import cluster_bootstrap_mean                           # noqa: E402
from run_arms import (MODEL, RETIRED, position_metrics,            # noqa: E402
                      reply_logprobs)
from adapters import GenericMlpAdapter, install_adapters           # noqa: E402

import scripts.train.memory as mem                                 # noqa: E402

REMNANT_BYTES_PER_LAYER = 4_939_872


class ZeroMlp(torch.nn.Module):
    """Exact deletion of an MLP sub-block: contributes nothing, holds nothing."""

    def forward(self, x):
        return torch.zeros_like(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("lora", "adapter"))
    ap.add_argument("--artifact", required=True,
                    help="adapter directory (lora) or .pt file (adapter)")
    ap.add_argument("--arm-key", default=None)
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--prompts", type=int, default=512)
    ap.add_argument("--skip-prompts", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds_path = REPO / DATASETS[args.dataset]
    if "SEALED" in ds_path.name:
        raise SystemExit("refusing to evaluate a sealed set")

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False

    prompts = load_prompts(ds_path, tokenizer, args.prompts, args.max_len,
                           skip=args.skip_prompts)
    if not prompts:
        raise SystemExit("no usable prompts loaded")

    # -- dense reference FIRST, from the pristine checkpoint ---------------
    t0 = time.time()
    dense_ref = []
    for p in prompts:
        dense_ref.append(reply_logprobs(model, p["ids"].to(device),
                                        p["boundary"]).to(torch.float16).cpu())
    print("dense reference cached in " + format(time.time() - t0, ".1f") + "s")

    # -- retire the four layers -------------------------------------------
    for l in RETIRED:
        model.model.layers[l].mlp = ZeroMlp()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    art = REPO / args.artifact
    meta: dict = {}
    if args.kind == "lora":
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        man = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
        meta = {k: v for k, v in man.items()
                if k not in ("loss_curve", "target_modules", "environment")}
        model = get_peft_model(model, LoraConfig(
            r=man["rank"], lora_alpha=man["alpha"], lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=list(man["target_modules"])))
        state = torch.load(art / "adapter.pt", map_location="cpu",
                           weights_only=True)
        set_peft_model_state_dict(model, {k: v.to(torch.float32)
                                          for k, v in state.items()})
        model = model.to(device).eval()
        n_param = man["trainable_parameters"]
        n_bytes = man["stored_bytes_bf16"]
    else:
        blob = torch.load(art, map_location="cpu", weights_only=False)
        meta = {k: v for k, v in blob.get("manifest", {}).items()
                if k not in ("loss_curve", "environment")}
        install_adapters(model, RETIRED, blob["hidden"], blob["bottleneck"],
                         state=blob["state"], device=device,
                         dtype=torch.bfloat16)
        model = model.eval()
        n_param = blob["manifest"]["trainable_parameters"]
        n_bytes = blob["manifest"]["stored_bytes_bf16"]

    budget = len(RETIRED) * REMNANT_BYTES_PER_LAYER
    key = args.arm_key or ("H-lora" if args.kind == "lora" else "I-adapter")
    print(key + ":  params " + format(n_param, ",") + "   stored "
          + format(n_bytes, ",") + " B (" + format(n_bytes / (1 << 20), ".3f")
          + " MiB)   = " + format(100 * n_bytes / budget, ".1f")
          + "% of the remnant budget\n")

    cols: dict[str, list] = {}
    clusters: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        for pi, p in enumerate(prompts):
            ids = p["ids"].to(device)
            b = p["boundary"]
            gold = ids[0, b:]
            d_lp = dense_ref[pi].to(device).float()
            a_lp = torch.log_softmax(
                model(ids).logits[0, b - 1:-1].float(), dim=-1)
            for k, v in position_metrics(d_lp, a_lp, gold).items():
                cols.setdefault(k, []).append(v.cpu())
            clusters.extend([pi] * int(gold.numel()))
            if (pi + 1) % 128 == 0:
                print("    " + str(pi + 1) + "/" + str(len(prompts)) + "  "
                      + format(time.time() - t0, ".0f") + "s", flush=True)

    flat = {k: torch.cat(v).to(torch.float64) for k, v in cols.items()}
    flat["d_nll_gold"] = flat["nll_gold"] - flat["nll_gold_dense"]
    boot = lambda n: cluster_bootstrap_mean(flat[n], clusters, n_boot=args.n_boot)
    entry = {
        "arm": key, "label": meta.get("tag", key), "kind": args.kind,
        "artifact": str(args.artifact),
        "n_positions": int(flat["kl"].numel()), "n_prompts": len(prompts),
        "kl": boot("kl"), "gold_top1": boot("gold_top1"),
        "nll_gold": boot("nll_gold"), "d_nll_gold": boot("d_nll_gold"),
        "agree": boot("agree"),
        "perplexity_gold": float(torch.exp(flat["nll_gold"].mean()).item()),
        "trainable_parameters": n_param,
        "stored_bytes_bf16": n_bytes,
        "remnant_budget_bytes": budget,
        "budget_match_fraction": n_bytes / budget,
        "training": meta,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("  {:<18s} KL {:>8.5f} [{:>7.5f},{:>7.5f}]  gold {:>6.4f}  ppl {:>8.3f}"
          .format(key, entry["kl"]["point"], entry["kl"]["lo"],
                  entry["kl"]["hi"], entry["gold_top1"]["point"],
                  entry["perplexity_gold"]))

    out = REPO / (args.out or ("experiments/sgqr_v2/results/recovery_"
                               + key + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "campaign": "sgqr_v2", "endpoint": "teacher_forced_paired",
        "retired_layers": list(RETIRED), "dataset": args.dataset,
        "prompts": len(prompts), "skip_prompts": args.skip_prompts,
        "max_len": args.max_len, "n_boot": args.n_boot,
        "arms": {key: entry}, "environment": collect(),
    }, indent=2) + "\n", encoding="utf-8")
    print("wrote " + str(out.relative_to(REPO)))


if __name__ == "__main__":
    main()
