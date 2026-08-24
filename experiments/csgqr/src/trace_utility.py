#!/usr/bin/env python3
"""Generate per-token confidence and marginal-utility labels for one layer.

What this produces
------------------
One record per reply position, carrying everything the routing analyses need and
nothing vocabulary-sized:

* the four confidence features at the pre-remnant boundary;
* ``kl_delete``, ``kl_remnant`` and the whole-remnant utility
  ``U = KL(p^T||p^0) - KL(p^T||p^Q)``;
* per-group utilities ``U_g`` (one float per group);
* top-1 ids under the dense, deleted and remnant paths, so top-token agreement
  and the "confidently wrong" analysis can be done without re-running anything;
* optional sampled group-pair utilities, for the interaction test.

One useful economy, and it is exact rather than an approximation
---------------------------------------------------------------
Only layer ``j``'s MLP is modified, so layers ``0..j-1`` are untouched in every
arm. The pre-remnant residual ``h_{j,t}`` depends only on those layers applied to
positions ``<= t``, and is therefore **identical across the dense, deleted and
remnant arms**. Confidence is a property of the surviving prefix, not of the arm,
so it is computed once from a single forward pass. ``--verify-invariance`` checks
this rather than assuming it.

Cost is ``3 + n_groups`` forward passes per prompt (dense, deleted, full remnant,
and one per group), plus two per sampled pair.

Sealed sets are never read: ``--dataset`` accepts only development splits.
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

from confidence import (AlignmentLens, ConfidenceHead,           # noqa: E402
                        features_from_logits)
from envinfo import collect, require_cuda                       # noqa: E402
from measure_layers import ARMS, DATASETS, load_prompts         # noqa: E402
from quant import QuantSpec                                     # noqa: E402
from router import RoutingPolicy                                # noqa: E402
from surgery import install_csgqr                               # noqa: E402

import scripts.train.memory as mem                              # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"


@torch.no_grad()
def logprobs_at(model, ids: torch.Tensor, boundary: int) -> torch.Tensor:
    logits = model(ids).logits[0, boundary - 1:-1]
    return torch.log_softmax(logits.float(), dim=-1)


def kl_rows(dense_lp: torch.Tensor, other_lp: torch.Tensor) -> torch.Tensor:
    p = dense_lp.exp()
    return (p * (dense_lp - other_lp)).sum(dim=-1)


@torch.no_grad()
def capture_residual(model, layer: int, ids: torch.Tensor,
                     boundary: int) -> torch.Tensor:
    """The pre-remnant residual at the reply positions.

    Taken as the input to ``post_attention_layernorm``, which is the tensor the
    decoder layer adds the MLP output back onto -- the same point
    ``surgery.py`` hooks.
    """
    grabbed: list[torch.Tensor] = []
    handle = model.model.layers[layer].post_attention_layernorm \
        .register_forward_pre_hook(lambda _m, args: grabbed.append(args[0]))
    try:
        model(ids)
    finally:
        handle.remove()
    return grabbed[0][0, boundary - 1:-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--arm", default="int4", choices=sorted(ARMS))
    ap.add_argument("--lens-rank", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=0,
                    help="number of sampled group pairs for the interaction test")
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--verify-invariance", action="store_true",
                    help="check that the pre-remnant residual is arm-independent")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(REPO / MODEL,
                                                 dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False

    prompts = load_prompts(REPO / DATASETS[args.dataset], tokenizer,
                           args.prompts, args.max_len)
    if not prompts:
        raise SystemExit("no usable prompts loaded")
    n_groups = args.n_groups
    spec = ARMS[args.arm]

    head = ConfidenceHead(final_norm=model.model.norm, lm_head=model.lm_head,
                          lens=AlignmentLens(model.config.hidden_size,
                                             args.lens_rank).to(device),
                          vocab_size=int(model.config.vocab_size))

    pair_list: list[tuple[int, int]] = []
    if args.pairs:
        g = torch.Generator().manual_seed(args.seed)
        seen = set()
        while len(pair_list) < args.pairs:
            a, b = torch.randint(0, n_groups, (2,), generator=g).tolist()
            if a != b and (min(a, b), max(a, b)) not in seen:
                seen.add((min(a, b), max(a, b)))
                pair_list.append((min(a, b), max(a, b)))

    print("layer " + str(args.layer) + "   arm " + args.arm + " (" + spec.name()
          + ")   " + str(n_groups) + " groups   lens_rank " + str(args.lens_rank))
    print("dataset " + args.dataset + "   " + str(len(prompts)) + " prompts, "
          + str(sum(p["reply_tokens"] for p in prompts)) + " reply positions")
    print("forwards per prompt: " + str(3 + n_groups + 2 * len(pair_list)))

    started = time.time()
    records: list[dict] = []
    invariance_max = 0.0

    for pi, p in enumerate(prompts):
        ids = p["ids"].to(device)
        b = p["boundary"]

        dense_lp = logprobs_at(model, ids, b)
        residual = capture_residual(model, args.layer, ids, b)
        # no_grad matters here: lm_head's weights require grad, so without it the
        # projection builds a graph over a [positions, 32768] tensor and holds it
        # for the whole prompt.
        with torch.no_grad():
            feats = features_from_logits(head.logits(residual), head.vocab_size)

        # -- deleted path
        inst = install_csgqr(model, args.layer,
                             policy=RoutingPolicy(mode="none", n_groups=1),
                             n_groups=1, keep_donor_resident=True)
        del_lp = logprobs_at(model, ids, b)
        if args.verify_invariance:
            r2 = capture_residual(model, args.layer, ids, b)
            invariance_max = max(invariance_max,
                                 (r2 - residual).abs().max().item())
        inst.remove()

        kl_del = kl_rows(dense_lp, del_lp)

        # -- remnant paths
        # Installed ONCE and the policy swapped between forwards. Re-installing
        # per configuration would re-quantize the same three matrices 27 times
        # per prompt, and -- worse than being slow -- would make each group's
        # measurement depend on a freshly built remnant rather than on exactly
        # the same packed weights the other groups were measured against.
        inst = install_csgqr(model, args.layer, spec=spec, n_groups=n_groups,
                             policy=RoutingPolicy(mode="all", n_groups=n_groups),
                             keep_donor_resident=True)
        module = inst.module

        def subset_policy(groups: list[int]) -> RoutingPolicy:
            rest = [k for k in range(n_groups) if k not in groups]
            return RoutingPolicy(mode="fixed_fraction",
                                 fraction=len(groups) / n_groups,
                                 selector="static_topk", n_groups=n_groups,
                                 ranking=list(groups) + rest)

        module.policy = RoutingPolicy(mode="all", n_groups=n_groups)
        kl_full = kl_rows(dense_lp, logprobs_at(model, ids, b))


        per_group_kl = []
        for g in range(n_groups):
            module.policy = subset_policy([g])
            per_group_kl.append(kl_rows(dense_lp, logprobs_at(model, ids, b)))

        pair_kl = {}
        for (a, bb) in pair_list:
            module.policy = subset_policy([a, bb])
            pair_kl[(a, bb)] = kl_rows(dense_lp, logprobs_at(model, ids, b))

        module.policy = RoutingPolicy(mode="all", n_groups=n_groups)
        full_lp = logprobs_at(model, ids, b)
        inst.remove()

        dense_top1 = dense_lp.argmax(-1)
        del_top1 = del_lp.argmax(-1)
        full_top1 = full_lp.argmax(-1)
        gold = ids[0, b:]

        for t in range(dense_lp.shape[0]):
            rec = {
                "prompt_index": pi,
                "prompt_id": p["id"],
                "position": t,
                "gold_token": int(gold[t].item()),
                "max_prob": float(feats["max_prob"][t]),
                "neg_entropy": float(feats["neg_entropy"][t]),
                "prob_margin": float(feats["prob_margin"][t]),
                "logit_margin": float(feats["logit_margin"][t]),
                "lens_top1": int(feats["top1"][t]),
                "kl_delete": float(kl_del[t]),
                "kl_remnant": float(kl_full[t]),
                "utility": float(kl_del[t] - kl_full[t]),
                "dense_top1": int(dense_top1[t]),
                "deleted_top1": int(del_top1[t]),
                "remnant_top1": int(full_top1[t]),
                "group_utility": [float(kl_del[t] - per_group_kl[g][t])
                                  for g in range(n_groups)],
            }
            if pair_list:
                rec["pair_utility"] = {
                    str(a) + "," + str(bb): float(kl_del[t] - pair_kl[(a, bb)][t])
                    for (a, bb) in pair_list}
            records.append(rec)

        if (pi + 1) % 8 == 0:
            print("  " + str(pi + 1) + "/" + str(len(prompts)) + "  "
                  + format(time.time() - started, ".0f") + "s", flush=True)

    tag = ("L" + str(args.layer) + "_" + args.arm + "_g" + str(n_groups)
           + "_" + args.dataset)
    out_dir = REPO / (args.out or ("experiments/csgqr/results/trace_" + tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    manifest = {
        "layer": args.layer, "arm": args.arm, "spec": spec.name(),
        "n_groups": n_groups, "dataset": args.dataset,
        "prompts": len(prompts), "positions": len(records),
        "lens_rank": args.lens_rank, "seed": args.seed,
        "pairs": [list(x) for x in pair_list],
        "residual_invariance_max_abs": (invariance_max if args.verify_invariance
                                        else None),
        "elapsed_s": round(time.time() - started, 1),
        "environment": collect(),
    }
    (out_dir / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    u = torch.tensor([r["utility"] for r in records], dtype=torch.float64)
    print("\n" + str(len(records)) + " positions in "
          + format(time.time() - started, ".0f") + "s")
    print("  mean KL(dense||delete)  " + format(
        sum(r["kl_delete"] for r in records) / len(records), ".5f"))
    print("  mean KL(dense||remnant) " + format(
        sum(r["kl_remnant"] for r in records) / len(records), ".5f"))
    print("  mean utility            " + format(u.mean().item(), ".5f"))
    print("  P(utility > 0)          " + format((u > 0).double().mean().item(), ".4f"))
    if args.verify_invariance:
        print("  residual invariance max |delta| " + format(invariance_max, ".3e")
              + "   (0 confirms confidence is arm-independent)")
    print("wrote " + str(out_dir.relative_to(REPO)))


if __name__ == "__main__":
    main()
