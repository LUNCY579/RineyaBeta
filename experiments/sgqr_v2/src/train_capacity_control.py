#!/usr/bin/env python3
"""The capacity control: train LoRA on the four-layer-pruned model, at matched bytes.

This is the experiment the whole campaign turns on
--------------------------------------------------
The alternative explanation for CSGQR-v1's headline result is simple and
plausible: **the remnant helps because it adds a few megabytes of parameters to a
damaged network, not because those particular weights are the deleted ones.**
CSGQR-v1 could only rule out a *wrong frozen donor*. It never ran a trained
same-byte baseline, so "beats generic capacity" was never tested.

This script trains that baseline, and it is written to be as generous to the
baseline as the budget allows, because a strawman here would invalidate the
campaign's central claim. Specifically:

* the byte budget is matched to the remnants **in stored bytes**, not parameter
  count, since the formats differ (packed uint8 + fp16 scales versus bf16 dense);
* rank is swept **past** the matched point, up to 175% of the remnant budget --
  if LoRA needs more bytes than the remnant to win, that is a finding, not a
  disqualification;
* the learning rate is the one the repository's own corpus README already
  validated for this checkpoint, and it is swept;
* the loss curve is written out, so undertraining is visible rather than
  arguable.

The asymmetry, stated rather than hidden
----------------------------------------
The remnant is **frozen and untrained**. LoRA is **trained**, and sees task data
the remnant never sees. That favours LoRA. It is the honest form of the question:
does free, untrained, exactly-preserved computation beat trained generic capacity
at the same storage cost? Training cost is reported so the asymmetry is priced.

Target-module policy
--------------------
The preregistration fixes LoRA to the seven projections of the **24 surviving
layers**. The retired layers' MLPs no longer exist (their projections are deleted
from the module tree), and their attention is deliberately left untouched so that
"generic capacity" means capacity in the surviving network rather than capacity
poured back into the same four slots. ``--include-retired-attn`` runs the other
policy as a sensitivity check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CSGQR_SRC = REPO / "experiments" / "csgqr" / "src"
TRAIN_SRC = REPO / "scripts" / "train"
for _p in (str(HERE), str(CSGQR_SRC), str(TRAIN_SRC), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset as ds                                               # noqa: E402
import memory as mem                                               # noqa: E402
from envinfo import collect, require_cuda                          # noqa: E402
from quant import QuantSpec                                        # noqa: E402
from multilayer import RetirementPlan, install_sgqr                # noqa: E402
from adapters import (GenericMlpAdapter, adapter_state,            # noqa: E402
                      bottleneck_for_budget, install_adapters)

MODEL = "artifacts/rineya-core-en32k-28L"
RETIRED = (13, 14, 24, 25)
#: measured, not estimated: analytic byte model and memory_allocated agree exactly
REMNANT_BYTES_PER_LAYER = 4_939_872
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")


def lora_target_modules(n_layers: int, retired, include_retired_attn: bool):
    """Explicit full module names, so the policy is auditable rather than implied.

    peft's short-name matching (`["q_proj", ...]`) would silently include the
    retired layers' attention, which is a different experiment. Naming every
    module makes the choice inspectable in the run manifest.
    """
    names = []
    for l in range(n_layers):
        if l in retired:
            if include_retired_attn:
                for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    names.append("model.layers." + str(l) + ".self_attn." + p)
            continue
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append("model.layers." + str(l) + ".self_attn." + p)
        for p in ("gate_proj", "up_proj", "down_proj"):
            names.append("model.layers." + str(l) + ".mlp." + p)
    return names


def adapter_bytes(model, dtype_bytes: int = 2) -> tuple[int, int]:
    """(trainable parameters, stored bytes at ``dtype_bytes`` per parameter).

    Reported at the *storage* dtype rather than the training dtype: LoRA is
    trained in fp32 for optimiser stability but would be shipped in bf16, which
    is what the remnant's packed bytes must be compared against.
    """
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n, n * dtype_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="lora", choices=("lora", "adapter"),
                    help="lora = generic capacity elsewhere in the surviving "
                         "network (arm H); adapter = a learned-from-scratch MLP "
                         "at the retired layer's own site (arm I)")
    ap.add_argument("--rank", type=int, default=18,
                    help="18 is the byte-matched rank (98.5%% of 4 remnants)")
    ap.add_argument("--alpha", type=int, default=None, help="default 2*rank")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=400,
                    help="optimizer steps; the loss curve is written so that "
                         "undertraining is visible rather than arguable")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mix", default="artifacts/heal_mixes/B0/heal_mix_b0.jsonl")
    ap.add_argument("--rows", type=int, default=0,
                    help="cap training rows (0 = all); rows are taken in a fixed "
                         "seeded permutation so a cap is a sample, not a prefix")
    ap.add_argument("--include-retired-attn", action="store_true")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--out-dir", default="artifacts/sgqr_v2_adapters")
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else 2 * args.rank

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False

    # -- retire the four layers, and actually free their weights -----------
    plan = RetirementPlan(layers=RETIRED, spec=QuantSpec(4, 128, False))
    inst = install_sgqr(model, plan, allocation={l: [] for l in RETIRED},
                        keep_donor_resident=False)
    for l in RETIRED:                       # drop the remnant too: this arm has none
        model.model.layers[l].mlp = _ZeroMlp()
    n_layers = model.config.num_hidden_layers

    budget = len(RETIRED) * REMNANT_BYTES_PER_LAYER
    targets: list[str] = []
    adapters: dict = {}
    bottleneck = 0
    if args.kind == "lora":
        targets = lora_target_modules(n_layers, set(RETIRED),
                                      args.include_retired_attn)
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=targets))
        model.enable_input_require_grads()
    else:
        # Arm I: a learned MLP at the retired site. Freeze the base entirely so
        # the only trainable bytes are the adapters -- otherwise this would be
        # full fine-tuning wearing an adapter's name.
        hidden = int(model.config.hidden_size)
        bottleneck = bottleneck_for_budget(REMNANT_BYTES_PER_LAYER, hidden)
        for p in model.parameters():
            p.requires_grad_(False)
        adapters = install_adapters(model, RETIRED, hidden, bottleneck,
                                    device="cpu", dtype=torch.bfloat16)
        for a in adapters.values():
            for p in a.parameters():
                p.requires_grad_(True)
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)

    n_param, n_bytes = adapter_bytes(model)
    print("four-layer-pruned model, " + args.kind + " capacity control")
    print("  retired layers      " + str(list(RETIRED)))
    if args.kind == "lora":
        print("  target modules      " + str(len(targets))
              + ("  (incl. retired attention)" if args.include_retired_attn else ""))
        print("  rank / alpha        " + str(args.rank) + " / " + str(alpha))
    else:
        print("  bottleneck          " + str(bottleneck)
              + "   (original d_ff 3072, so "
              + format(100 * bottleneck / 3072, ".1f") + "% of the width)")
    print("  trainable params    " + format(n_param, ","))
    print("  stored bytes (bf16) " + format(n_bytes, ",") + "  = "
          + format(n_bytes / (1 << 20), ".3f") + " MiB")
    print("  remnant budget      " + format(budget, ",") + "  = "
          + format(budget / (1 << 20), ".3f") + " MiB")
    print("  budget match        " + format(100 * n_bytes / budget, ".1f") + "%\n")

    # -- data --------------------------------------------------------------
    data = ds.build(REPO / args.mix, tokenizer, max_len=args.max_len,
                    cache_dir=REPO / "artifacts" / "tok_cache",
                    tokenizer_dir=REPO / MODEL, verbose=True)
    n_rows = len(data.prompt_lens)
    order = np.random.RandomState(args.seed).permutation(n_rows)
    if args.rows:
        order = order[:args.rows]
    print("training rows " + format(len(order), ",") + " of "
          + format(n_rows, ",") + "   max_len " + str(args.max_len))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    pad_id = tokenizer.pad_token_id
    curve = []
    cursor = 0
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        lr = (args.lr * step / max(args.warmup, 1) if step <= args.warmup else
              args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup)
                                            / max(args.steps - args.warmup, 1))))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total_tokens, total_loss = 0, 0.0
        for _ in range(args.grad_accum):
            idx = [int(order[(cursor + j) % len(order)])
                   for j in range(args.micro_batch)]
            cursor += args.micro_batch
            ids_np, att_np, lab_np = ds.collate(data, idx, pad_id, args.max_len)
            ids = torch.from_numpy(ids_np).to(device)
            att = torch.from_numpy(att_np).to(device)
            lab = torch.from_numpy(lab_np).to(device)
            logits = model(input_ids=ids, attention_mask=att).logits
            sl = logits[:, :-1, :].float()
            tl = lab[:, 1:]
            loss_sum = F.cross_entropy(sl.reshape(-1, sl.size(-1)),
                                       tl.reshape(-1),
                                       ignore_index=ds.IGNORE_INDEX,
                                       reduction="sum")
            counted = int((tl != ds.IGNORE_INDEX).sum().item())
            (loss_sum / max(counted, 1) / args.grad_accum).backward()
            total_loss += float(loss_sum.detach())
            total_tokens += counted
            del logits, sl, loss_sum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        mean_loss = total_loss / max(total_tokens, 1)
        curve.append({"step": step, "lr": lr, "loss": mean_loss,
                      "tokens": total_tokens})
        if step % 20 == 0 or step == 1:
            print("  step {:>4d}/{:<4d}  loss {:>8.4f}  lr {:.2e}  {:>6.0f}s"
                  .format(step, args.steps, mean_loss, lr, time.time() - t0),
                  flush=True)

    tag = ((("r" + str(args.rank)) if args.kind == "lora"
            else ("m" + str(bottleneck)))
           + "_lr" + format(args.lr, ".0e") + "_s" + str(args.steps)
           + ("_attn" if args.include_retired_attn else ""))
    out = REPO / args.out_dir / (args.kind + "_" + tag)
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "lora":
        state = get_peft_model_state_dict(model)
        torch.save({k: v.to(torch.bfloat16) for k, v in state.items()},
                   out / "adapter.pt")
    else:
        torch.save({"kind": "adapter", "hidden": int(model.config.hidden_size)
                    if hasattr(model, "config") else 1024,
                    "bottleneck": bottleneck,
                    "state": adapter_state(adapters),
                    "manifest": {"trainable_parameters": n_param,
                                 "stored_bytes_bf16": n_bytes,
                                 "bottleneck": bottleneck, "tag": tag}},
                   out / "adapter.pt")
    on_disk = (out / "adapter.pt").stat().st_size

    manifest = {
        "arm": "H-lora" if args.kind == "lora" else "I-adapter",
        "kind": args.kind, "bottleneck": bottleneck, "tag": tag,
        "retired_layers": list(RETIRED),
        "rank": args.rank, "alpha": alpha, "lr": args.lr,
        "steps": args.steps, "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum, "max_len": args.max_len,
        "rows_used": int(len(order)), "rows_available": int(n_rows),
        "mix": args.mix, "mix_sha256": data.mix_sha256,
        "include_retired_attn": args.include_retired_attn,
        "target_module_count": len(targets),
        "target_modules": targets,
        "trainable_parameters": n_param,
        "stored_bytes_bf16": n_bytes,
        "stored_bytes_on_disk": on_disk,
        "remnant_budget_bytes": budget,
        "budget_match_fraction": n_bytes / budget,
        "final_loss": curve[-1]["loss"] if curve else None,
        "first_loss": curve[0]["loss"] if curve else None,
        "loss_curve": curve,
        "train_seconds": round(time.time() - t0, 1),
        "seed": args.seed,
        "environment": collect(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print("\nfinal loss " + format(curve[-1]["loss"], ".4f")
          + "   (from " + format(curve[0]["loss"], ".4f") + ")")
    print("adapter on disk " + format(on_disk, ",") + " B")
    print("wrote " + str(out.relative_to(REPO)))


class _ZeroMlp(torch.nn.Module):
    """An MLP sub-block that contributes nothing. Exact deletion, no remnant."""

    def forward(self, x):
        return torch.zeros_like(x)


if __name__ == "__main__":
    main()
