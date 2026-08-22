#!/usr/bin/env python3
"""P0.1 -- is there any sparsity in this SwiGLU FFN to exploit?

THE KILL TEST. The stabilizer matrix proposes skipping FFN neurons that will not
contribute. That is only possible if a small fraction of neurons carries most of
the layer output. For ReLU models that is famously true. This checkpoint uses
SiLU (`hidden_act: silu`, SwiGLU MLP), and the relufication literature reports
that SiLU/GELU families concentrate far less. So this probe can end the project
in an afternoon, which is exactly why it runs first.

WHAT IS MEASURED. For a SwiGLU block,

    act  = silu(W_gate x) * (W_up x)        [3072]
    out  = W_down @ act                     [1024]

"firing neuron j" means computing act[j] and adding act[j] * W_down[:, j] to the
output. So the question is what fraction of ||out|| survives if only the top-k
neurons are kept:

    rho(k) = || W_down @ topk_k(act) ||  /  || W_down @ act ||

TWO RANKINGS, because the obvious one is wrong. Ranking by |act[j]| ignores that
a neuron's contribution is act[j] * W_down[:, j], so a large activation
multiplying a small column contributes little. We measure both:

    by_act    rank on |act[j]|                        -- what a predictor sees
    by_contrib rank on |act[j]| * ||W_down[:, j]||     -- what actually matters

The gap between them is itself a finding: if `by_contrib` concentrates much
better than `by_act`, then any selector must know the column norms, which are
static and free, and a purely activation-based predictor is aiming at the wrong
target.

A RANDOM CONTROL is included at every k. Concentration means nothing without it:
with 3072 neurons and a heavy-tailed norm distribution, a random quarter already
recovers a substantial share, and the KV program has already been caught once by
an "obviously concentrated" statistic that a random baseline matched.

PRE-DECLARED THRESHOLDS (docs/EXPERIMENT_DESIGN_STABILIZER.md section 5):
    rho(25%) >= 0.95  -> strong substrate, proceed
    0.85 <= rho < 0.95 -> weak, proceed only if P0.2 shows a speed win
    rho(25%) < 0.85   -> KILL

Usage:
    python src/concentration_mlp.py
    python src/concentration_mlp.py --prompts 16 --dataset rag_dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

MODEL = "artifacts/rineya-core-en32k-28L"
FRACS = (0.0625, 0.125, 0.25, 0.375, 0.50, 0.75)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--dataset", default="rag_dev")
    ap.add_argument("--cap-mib", type=float, default=6000.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "concentration_mlp.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    force_repeat_kv(verbose=False)
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)

    tok = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()
    layers = model.model.layers
    L = len(layers)

    # Capture the SwiGLU intermediate. The hook sits on down_proj's INPUT, which
    # is exactly `act` -- no re-derivation, no chance of reconstructing it wrong.
    CAP: dict[int, torch.Tensor] = {}

    def mk(i):
        def hook(mod, inp, out):
            CAP[i] = inp[0].detach()
        return hook

    for i, lyr in enumerate(layers):
        lyr.mlp.down_proj.register_forward_hook(mk(i))

    # Static per-neuron column norms of W_down: free to compute, and the correct
    # weighting for "how much does this neuron move the output".
    colnorm = [layers[i].mlp.down_proj.weight.float().norm(dim=0) for i in range(L)]

    path = REPO / "artifacts" / "eval_sets" / f"{args.dataset}.jsonl"
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            recs.append([m for m in d["messages"] if m["role"] != "assistant"])
    recs = recs[:: max(1, len(recs) // args.prompts)][: args.prompts]

    g = torch.Generator(device="cpu").manual_seed(7)
    acc = {r: {f: [] for f in FRACS} for r in ("by_act", "by_contrib", "random")}
    per_layer = {r: {f: [[] for _ in range(L)] for f in FRACS} for r in
                 ("by_act", "by_contrib", "random")}
    n_tokens = 0

    for msgs in recs:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        CAP.clear()
        model(ids, use_cache=False)
        n_tokens += ids.shape[-1]

        for li in range(L):
            a = CAP[li][0].float()                       # [seq, 3072]
            W = layers[li].mlp.down_proj.weight.float()  # [1024, 3072]
            full = a @ W.T                               # [seq, 1024]
            fn = full.norm(dim=-1).clamp_min(1e-9)
            n_neurons = a.shape[-1]

            score = {
                "by_act": a.abs(),
                "by_contrib": a.abs() * colnorm[li].unsqueeze(0),
            }
            for frac in FRACS:
                k = max(1, int(round(frac * n_neurons)))
                for rank in ("by_act", "by_contrib", "random"):
                    if rank == "random":
                        idx = torch.stack([
                            torch.randperm(n_neurons, generator=g)[:k]
                            for _ in range(a.shape[0])]).to(a.device)
                    else:
                        idx = score[rank].topk(k, dim=-1).indices
                    kept = torch.zeros_like(a).scatter_(1, idx, a.gather(1, idx))
                    r = ((kept @ W.T).norm(dim=-1) / fn).mean().item()
                    acc[rank][frac].append(r)
                    per_layer[rank][frac][li].append(r)
        del CAP
        CAP = {}
        for i, lyr in enumerate(layers):
            pass
        torch.cuda.empty_cache()

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"P0.1 -- SwiGLU activation concentration, {len(recs)} prompts, "
          f"{n_tokens} tokens, {L} layers, 3072 neurons\n")
    print(f"{'keep':>7}{'by_act':>10}{'by_contrib':>13}{'random':>10}"
          f"{'contrib-random':>17}")
    rows = []
    for frac in FRACS:
        a_, c_, r_ = (mean(acc[k][frac]) for k in ("by_act", "by_contrib", "random"))
        rows.append({"keep_frac": frac, "by_act": a_, "by_contrib": c_, "random": r_})
        print(f"{frac*100:>6.1f}%{a_:>10.4f}{c_:>13.4f}{r_:>10.4f}{c_-r_:>17.4f}")

    key = next(r for r in rows if abs(r["keep_frac"] - 0.25) < 1e-9)
    rho25 = key["by_contrib"]
    verdict = ("STRONG -- proceed" if rho25 >= 0.95 else
               "WEAK -- proceed only if P0.2 shows a speed win" if rho25 >= 0.85 else
               "KILL -- SiLU does not concentrate enough")
    print(f"\nrho(25%) by contribution = {rho25:.4f}   ->   {verdict}")
    print(f"random control at 25%     = {key['random']:.4f}"
          f"   (margin {rho25 - key['random']:+.4f})")
    if rho25 - key["random"] < 0.05:
        print("\nNOTE: the margin over random is small. Concentration that a random\n"
              "subset matches is not concentration -- it is the norm distribution.")

    # worst layer matters more than the mean: one un-skippable layer bounds the whole
    pl25 = [mean(per_layer["by_contrib"][0.25][li]) for li in range(L)]
    worst = min(range(L), key=lambda i: pl25[i])
    print(f"\nper-layer rho(25%): min {pl25[worst]:.4f} (layer {worst}), "
          f"max {max(pl25):.4f}, mean {mean(pl25):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "dataset": args.dataset, "prompts": len(recs),
         "tokens": n_tokens, "layers": L, "neurons": 3072,
         "rho25_by_contrib": rho25, "rho25_random": key["random"],
         "verdict": verdict, "results": rows,
         "per_layer_rho25_by_contrib": pl25}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
