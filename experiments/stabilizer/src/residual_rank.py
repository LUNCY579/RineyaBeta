#!/usr/bin/env python3
"""P0.4 -- can a low-rank term replace the neurons a static prune deletes?

The stabilizer matrix was proposed as a *selector*. P0.2 killed that on speed, and
a selector cannot compress anyway: choosing the top-k of 3,072 neurons per token
needs all 3,072 weight columns resident. P0.3's `static` arm was the permanent-prune
test -- "if this works, nothing else is needed" -- and it did not work, giving
rho = 0.7468 against the predictor's 0.9027.

This probe asks the compression question instead. Delete the low-firing neurons
*permanently*, then approximate what they contributed with a rank-r term:

    F = a @ W_down.T                     full FFN output      [T, 1024]
    K = a[:,S] @ W_down[:,S].T           kept, static set S   [T, 1024]
    Y = F - K                            what the prune threw away
    Y_r = best rank-r approximation of Y

Parameter economics, per layer (hidden 1024, intermediate 3072):

    full FFN            3 * 1024 * 3072 = 9,437,184
    keep fraction k     3 * 1024 * 3072k
    rank-r correction   r * (1024 + 1024) = 2048r

At k = 0.25 and r = 64 that is 2,490,368 against 9,437,184 -- saving 6.95M per
layer, 194.5M over 28 layers, which would take R2 from 474.0M to ~279.5M. Bigger
than the vocabulary trim. Break-even is r < 3,457, so rank is not the parameter
constraint; the only question is whether a small r suffices.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
Y_r here is the *oracle* rank-r approximation, computed by SVD of Y on the very
tokens being measured. A real correction is a fixed learned matrix that has never
seen these tokens, so it can only do worse. This is therefore a **ceiling**: if the
ceiling fails the idea is dead, and if it passes the idea is merely not-yet-refuted.
Stated up front so a passing number is not mistaken for a working method.

Pre-declared threshold (before measurement, per P5): the rank at 90% of Y's energy
must be <= 128, which costs 262,144 params/layer and still saves >180M overall.
Above that, the correction stops being cheap and the idea is not worth building.

Usage: ../../.venv/Scripts/python.exe residual_rank.py [--keep 0.25] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
MODEL = "artifacts/rineya-core-en32k-28L"
DATASET = "artifacts/eval_sets/rag_dev.jsonl"
RANKS = (16, 32, 64, 128, 256, 512)
ENERGY = (0.90, 0.95, 0.99)


def energy_rank(sv: torch.Tensor, frac: float) -> int:
    power = sv.double() ** 2
    return int(torch.searchsorted(torch.cumsum(power, 0) / power.sum(), frac).item()) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "residual_rank.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)  # another experiment may be using this machine

    tok = AutoTokenizer.from_pretrained(REPO / MODEL)
    # bf16 forward to match P0.1 exactly; all analysis in float32 as that script does.
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()
    layers = model.model.layers
    L = len(layers)

    recs = []
    with (REPO / DATASET).open(encoding="utf-8") as fh:
        for line in fh:
            if len(recs) >= args.prompts:
                break
            row = json.loads(line)
            msgs = [m for m in row["messages"] if m.get("role") != "assistant"]
            if msgs:
                recs.append(msgs)

    # Same capture point as concentration_mlp.py: down_proj's INPUT is the SwiGLU
    # intermediate, so no need to recompute gate/up.
    grabbed: dict[int, torch.Tensor] = {}

    def mk(i):
        def hook(mod, inp, out):
            grabbed[i] = inp[0].detach()[0].float()   # [T, 3072]
        return hook

    handles = [layers[i].mlp.down_proj.register_forward_hook(mk(i)) for i in range(L)]

    acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(L)}
    total_tokens = 0
    for msgs in recs:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)
        with torch.inference_mode():
            model(**ids)
        total_tokens += int(ids.input_ids.shape[1])
        for i in range(L):
            acts[i].append(grabbed[i].cpu())
    for h in handles:
        h.remove()

    print(f"model   : {MODEL}")
    print(f"prompts : {len(recs)}   tokens : {total_tokens}   layers : {L}")
    print(f"keep    : {args.keep:.0%}  (static per-layer set, chosen by mean |activation|)")
    print(f"device  : {args.device}\n")

    per_layer = []
    for li in range(L):
        a = torch.cat(acts[li], 0)                                    # [T, 3072]
        W = layers[li].mlp.down_proj.weight.float().cpu()              # [1024, 3072]
        n = a.shape[1]
        k = max(1, int(round(n * args.keep)))

        # Static set: the same rule P0.3 used for its prune arm -- ranked by how
        # strongly each neuron fires across the corpus, not per token.
        keep_idx = a.abs().mean(0).topk(k).indices
        mask = torch.zeros(n, dtype=torch.bool)
        mask[keep_idx] = True

        F = a @ W.T
        K = a[:, mask] @ W[:, mask].T
        Y = F - K
        nF = F.norm().clamp_min(1e-9)

        # rho as concentration_mlp.py defines it: per-token norm ratio, then mean.
        rho_static = ((K.norm(dim=-1) / F.norm(dim=-1).clamp_min(1e-9)).mean()).item()

        sv = torch.linalg.svdvals(Y)
        tail = torch.flip(torch.cumsum(torch.flip(sv.double() ** 2, [0]), 0), [0])
        entry = {"layer": li, "rho_static": rho_static,
                 "err_static": (Y.norm() / nF).item(),
                 "ranks": {str(int(e * 100)): energy_rank(sv, e) for e in ENERGY}}
        for r in RANKS:
            resid = float(tail[r].sqrt()) if r < len(sv) else 0.0
            entry[f"err_rank_{r}"] = resid / float(nF)
        per_layer.append(entry)
        if li % 7 == 0 or li == L - 1:
            print(f"  layer {li:>2d}: rho_static {rho_static:.4f}  "
                  f"rank@90% {entry['ranks']['90']:>4d}  "
                  f"err r=64 {entry['err_rank_64']:.4f}")

    def mean(key):
        return sum(e[key] for e in per_layer) / len(per_layer)

    r90 = sum(e["ranks"]["90"] for e in per_layer) / L
    print(f"\n=== averages over {L} layers ===")
    print(f"  rho_static (control; P0.3 reported 0.7468)   {mean('rho_static'):.4f}")
    print(f"  relative error with NO correction            {mean('err_static'):.4f}")
    print(f"\n  {'rank r':>7s} {'rel. error':>11s} {'params/layer':>13s} {'x28':>9s} "
          f"{'R2 -> ':>9s}")
    kept_params = 3 * 1024 * int(round(3072 * args.keep))
    for r in RANKS:
        p = kept_params + 2048 * r
        print(f"  {r:>7d} {mean(f'err_rank_{r}'):>11.4f} {p:>13,d} "
              f"{(9_437_184 - p) * 28 / 1e6:>8.1f}M {474.0 - (9_437_184 - p) * 28 / 1e6:>8.1f}M")

    print(f"\n  rank at 90% of the discarded energy: mean {r90:.0f}  "
          f"(threshold was <= 128)")
    print(f"  VERDICT: {'PASS' if r90 <= 128 else 'FAIL'} -- "
          + ("a cheap low-rank term can carry most of what the prune removes; "
             "next question is whether a *fixed* matrix gets close to this ceiling"
             if r90 <= 128 else
             "the discarded contribution is not low-rank, so the correction is not "
             "cheap and the idea does not survive its own kill-test"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": MODEL, "dataset": DATASET, "prompts": len(recs),
        "tokens": total_tokens, "layers": L, "keep": args.keep,
        "device": args.device, "ranks_probed": list(RANKS),
        "threshold_rank90": 128, "is_oracle_ceiling": True,
        "mean_rho_static": mean("rho_static"),
        "mean_err_static": mean("err_static"),
        "mean_rank90": r90,
        "mean_err_by_rank": {str(r): mean(f"err_rank_{r}") for r in RANKS},
        "per_layer": per_layer,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
