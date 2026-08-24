#!/usr/bin/env python3
"""Gate A: per-layer MLP deletion damage, and how much a frozen remnant recovers.

This is the experiment that decides whether the rest of the programme is worth
running. If an always-on INT4 remnant cannot recover a meaningful fraction of
the damage that deleting an MLP causes, then no routing policy over that remnant
can either -- routing can only ever spend *less* of it.

    Q_CSGQR  <=  Q_INT4-all      (approximately, absent survivor adaptation)

Metric
------
Full generation-and-score over 28 layers x 5 arms would be hours of GPU time, so
the scan uses a teacher-forced divergence proxy that needs one forward pass per
configuration:

    KL_arm = mean over reply positions of  KL( p_dense || p_arm )

measured at the positions where the model would actually be generating (the gold
assistant turn is teacher-forced, and prompt positions are excluded, because a
divergence averaged over prompt positions is dominated by tokens no arm ever has
to produce). Top-1 agreement with the dense model is reported alongside, because
KL and argmax-agreement can move independently and the quality evaluator only
ever sees the argmax.

    kl_recovery = (KL_delete - KL_arm) / KL_delete

is the fraction of the *divergence* damage removed. It is a proxy for the
Recovery Ratio, not the Recovery Ratio itself: that one is defined on the quality
metric and is computed by ``run_eval.py`` on the layers this scan selects. Where
``KL_delete`` is tiny the ratio is unstable, and the raw KL is reported next to
it so that instability is visible rather than hidden.

Why the FP arm is here
----------------------
``fp`` is the donor at full precision travelling the *remnant code path*. It
isolates two things at once: the cost of freezing (none -- the weights are the
originals) and the cost of the code path itself (cuBLAS picks different bf16 GEMM
tiles for some shapes, so this arm is not always bit-identical to dense). Any
apparent damage in the ``fp`` arm is the floor below which no quantization result
can be believed.
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

from envinfo import collect, require_cuda                     # noqa: E402
from quant import QuantSpec                                   # noqa: E402
from router import RoutingPolicy                              # noqa: E402
from surgery import install_csgqr                             # noqa: E402

import scripts.train.memory as mem                            # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
DATASETS = {"chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
            "rag_dev": "artifacts/eval_sets/rag_dev.jsonl"}

ARMS = {
    "fp": QuantSpec(16, 0, True),
    "int8": QuantSpec(8, 128, False),
    "int4": QuantSpec(4, 128, False),
    "int4sym": QuantSpec(4, 128, True),
    "int4b64": QuantSpec(4, 64, False),
}


def load_prompts(path: Path, tokenizer, limit: int, max_len: int,
                 skip: int = 0) -> list[dict]:
    """Templated prompt+reply pairs with the reply boundary recorded.

    The boundary is computed from the *prompt-only* rendering with
    ``add_generation_prompt=True``, which is exactly how ``run_gates.py`` builds
    its input, so "reply positions" here are the positions that arm would have
    been generating under the real protocol.

    ``skip`` discards the first ``skip`` *usable* prompts before collecting, so a
    caller can take a slice disjoint from an earlier run's. This is what keeps a
    probe's evaluation honest: the L20 trace was built from the first 96 usable
    ``chat_dev`` prompts and the probes were fitted on a split of those, so
    evaluating a probe-gated arm on prompts 0.. would score it partly on its own
    training data. Counting *usable* rather than raw lines matters -- the filters
    below reject records, so a raw-line offset would not line up with the earlier
    run's prompt indices.
    """
    out: list[dict] = []
    seen = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(out) >= limit:
            break
        rec = json.loads(line)
        msgs = [dict(m) for m in rec.get("messages") or []]
        if not msgs or msgs[-1].get("role") != "assistant":
            continue
        prompt_msgs, reply = msgs[:-1], str(msgs[-1].get("content", ""))
        if not reply.strip():
            continue
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        full_text = prompt_text + reply
        ids = tokenizer(full_text, return_tensors="pt").input_ids
        boundary = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        if ids.shape[1] > max_len or boundary >= ids.shape[1]:
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append({"id": rec.get("id"), "ids": ids, "boundary": boundary,
                    "reply_tokens": ids.shape[1] - boundary})
    return out


@torch.no_grad()
def reply_logprobs(model, ids: torch.Tensor, boundary: int) -> torch.Tensor:
    """Log-probs over the positions that predict the reply tokens.

    Position ``boundary-1`` predicts the first reply token, so the slice starts
    there. Returned in float32 on CPU: keeping these on the GPU across 28 layers
    of scanning would not fit, and bf16 log-probs over 32768 classes carry
    enough error to move the KL in the third decimal.
    """
    logits = model(ids).logits[0, boundary - 1:-1]
    return torch.log_softmax(logits.float(), dim=-1).cpu()


def kl_and_agreement(dense_lp: torch.Tensor,
                     arm_lp: torch.Tensor) -> tuple[float, float]:
    """``mean KL(dense || arm)`` in nats, and top-1 agreement rate."""
    p = dense_lp.exp()
    kl = (p * (dense_lp - arm_lp)).sum(dim=-1)
    agree = (dense_lp.argmax(-1) == arm_lp.argmax(-1)).float()
    return kl.mean().item(), agree.mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--layers", default="all",
                    help="'all' or a comma-separated list")
    ap.add_argument("--arms", default="fp,int8,int4")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--out", default="experiments/csgqr/results/layer_scan.json")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(REPO / MODEL,
                                                 dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False          # single forward pass; no cache needed
    n_layers = model.config.num_hidden_layers

    prompts = load_prompts(REPO / DATASETS[args.dataset], tokenizer,
                           args.prompts, args.max_len)
    if not prompts:
        raise SystemExit("no usable prompts loaded")
    positions = sum(p["reply_tokens"] for p in prompts)
    layers = (list(range(n_layers)) if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            raise SystemExit("unknown arm " + a + "; have " + str(sorted(ARMS)))

    print("dataset  " + args.dataset + "   " + str(len(prompts)) + " prompts, "
          + str(positions) + " reply positions")
    print("layers   " + str(layers))
    print("arms     delete, " + ", ".join(arm_names))
    print("scoring  mean KL(dense || arm) over reply positions + top-1 agreement\n")

    started = time.time()
    dense = [reply_logprobs(model, p["ids"].to(device), p["boundary"])
             for p in prompts]
    print("dense reference cached in " + format(time.time() - started, ".1f") + "s")

    rows: list[dict] = []
    for layer in layers:
        line = {"layer": layer}
        for arm in ["delete"] + arm_names:
            if arm == "delete":
                policy = RoutingPolicy(mode="none", n_groups=1)
                inst = install_csgqr(model, layer, policy=policy, n_groups=1,
                                     keep_donor_resident=True)
            else:
                policy = RoutingPolicy(mode="all", n_groups=args.n_groups)
                inst = install_csgqr(model, layer, spec=ARMS[arm],
                                     n_groups=args.n_groups, policy=policy,
                                     keep_donor_resident=True)
            kls, agrs = [], []
            for p, ref in zip(prompts, dense):
                lp = reply_logprobs(model, p["ids"].to(device), p["boundary"])
                kl, ag = kl_and_agreement(ref, lp)
                kls.append(kl)
                agrs.append(ag)
            inst.remove()
            # Token-weighted, so a long reply is not equal to a one-token one.
            w = torch.tensor([p["reply_tokens"] for p in prompts], dtype=torch.float64)
            kl = (torch.tensor(kls, dtype=torch.float64) * w).sum().item() / w.sum().item()
            ag = (torch.tensor(agrs, dtype=torch.float64) * w).sum().item() / w.sum().item()
            line[arm + "_kl"] = kl
            line[arm + "_agreement"] = ag

        kd = line["delete_kl"]
        for arm in arm_names:
            line[arm + "_kl_recovery"] = ((kd - line[arm + "_kl"]) / kd
                                          if kd > 0 else float("nan"))
        rows.append(line)
        print("  L{:<3d} delete KL {:>8.4f} agree {:>6.3f}".format(
            layer, line["delete_kl"], line["delete_agreement"])
            + "".join("   {} KL {:>7.4f} rec {:>7.3f}".format(
                a, line[a + "_kl"], line[a + "_kl_recovery"]) for a in arm_names),
            flush=True)

    ordered = sorted(rows, key=lambda r: r["delete_kl"])
    kls = [r["delete_kl"] for r in ordered]

    def pick(target: float) -> dict:
        return min(rows, key=lambda r: abs(r["delete_kl"] - target))

    lo, mid, hi = kls[0], kls[len(kls) // 2], kls[-1]
    candidates = {
        "low_sensitivity": pick(lo)["layer"],
        "moderate_sensitivity": pick(mid)["layer"],
        "high_sensitivity": pick(hi)["layer"],
    }

    out = {
        "dataset": args.dataset,
        "prompts": len(prompts),
        "reply_positions": positions,
        "max_len": args.max_len,
        "n_groups": args.n_groups,
        "arms": {a: ARMS[a].name() for a in arm_names},
        "metric": "mean KL(p_dense || p_arm) over reply positions, token-weighted",
        "rows": rows,
        "deletion_damage_order_ascending": [r["layer"] for r in ordered],
        "candidates": candidates,
        "elapsed_s": round(time.time() - started, 1),
        "environment": collect(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\ndeletion damage, ascending: " + str(out["deletion_damage_order_ascending"]))
    print("candidates: " + json.dumps(candidates))
    for arm in arm_names:
        vals = [r[arm + "_kl_recovery"] for r in rows]
        print("  " + arm + " KL recovery: min " + format(min(vals), ".3f")
              + "  median " + format(sorted(vals)[len(vals) // 2], ".3f")
              + "  max " + format(max(vals), ".3f"))
    print("\nwrote " + str(dest.relative_to(REPO)) + "  ("
          + format(time.time() - started, ".0f") + "s)")


if __name__ == "__main__":
    main()
