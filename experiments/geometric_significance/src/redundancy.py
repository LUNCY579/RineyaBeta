#!/usr/bin/env python3
"""Phase 0 for the geometric idea: is the selected key set actually redundant?

The geometry-aware proposal only pays if plain top-k selection is
REDUNDANCY-BLIND in a way that costs something. The argument is that if many
cached keys are near-duplicates and all score highly, top-k spends many slots on
one piece of information, while a distinct lower-scoring key gets evicted --
and the retained-mass bound cannot see this, because duplicated attention mass
counts fully toward rho.

That argument has a measurable premise. This measures it, before any policy is
designed, so the idea can be killed for the price of one forward pass (P7).

WHAT IS MEASURED, per layer and KV head, over the top-B keys chosen by the
existing significance policy:

  1. pairwise cosine similarity, against two controls -- a random-B subset of
     the same cache, and all n keys. If top-B is no more self-similar than a
     random subset, selection is not concentrating on a mode and there is
     nothing for a diversity term to fix.
  2. effective rank of the selected set, via the participation ratio of the
     Gram spectrum: exp(H(p)) where p_i = lambda_i / sum(lambda). B distinct
     directions gives ~B; heavy duplication gives much less. The ratio
     eff_rank / B is the headroom the geometric idea is claiming.
  3. cluster count at a cosine threshold. How many DISTINCT things are in the
     budget, as opposed to how many slots it occupies.

Keys are taken post-RoPE, which is what attention actually multiplies against,
so similarity here is similarity of attention behaviour rather than of raw
content. Note this makes distant positions look less similar than their content
alone would suggest -- an effect that works AGAINST finding redundancy, so a
positive result here is conservative.

Usage:
    python src/redundancy.py --cases 8 --budget 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
_KEYS: dict[int, torch.Tensor] = {}


def capture_keys() -> None:
    """Record post-RoPE, pre-repeat_kv keys per layer alongside the scores."""
    from transformers.integrations import sdpa_attention as sa
    original = sa.sdpa_attention_forward

    def wrapped(*args, **kw):
        # Signature-agnostic: transformers calls this with (module, query, key,
        # value, attention_mask, ...) positionally, and the count has changed
        # between versions. Take the first three positionally and pass the rest
        # through untouched.
        if len(args) >= 3:
            module, key = args[0], args[2]
            li = getattr(module, "layer_idx", None)
            if li is not None:
                _KEYS[li] = key.detach()
        return original(*args, **kw)

    sa.sdpa_attention_forward = wrapped
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
    except Exception:  # noqa: BLE001
        pass


def eff_rank(gram: torch.Tensor) -> float:
    """exp(entropy of the normalised Gram spectrum) -- 'how many directions'."""
    ev = torch.linalg.eigvalsh(gram.double()).clamp_min(0)
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    p = ev / tot
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def n_clusters(sim: torch.Tensor, tau: float) -> int:
    """Greedy cover: how many keys survive suppressing neighbours above tau."""
    n = sim.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=sim.device)
    count = 0
    order = torch.arange(n, device=sim.device)
    for i in order.tolist():
        if not alive[i]:
            continue
        count += 1
        alive &= sim[i] < tau
        alive[i] = False
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/redundancy.json"))
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    capture_keys()

    path = REPO / MODEL
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()

    prompts = []
    with (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                msgs = [m for m in rec["messages"] if m.get("role") != "assistant"]
                prompts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            if len(prompts) >= args.cases:
                break

    g = torch.Generator(device="cpu").manual_seed(0)
    acc = {k: [] for k in ("sim_top", "sim_rand", "sim_all", "er_top", "er_rand",
                           "clu_top", "clu_rand", "sim_ctop", "sim_crand",
                           "er_ctop", "er_crand", "clu_ctop", "clu_crand")}
    per_layer = {k: torch.zeros(model.config.num_hidden_layers) for k in acc}
    counts = torch.zeros(model.config.num_hidden_layers)

    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to(args.device)
        n = enc["input_ids"].shape[-1]
        if n <= args.budget:
            continue
        sig.reset_scores()
        _KEYS.clear()
        sig.set_recording(True, window=args.window)
        with torch.no_grad():
            model(**enc, use_cache=True, logits_to_keep=1)
        sig.set_recording(False)

        scores = sig.get_scores()
        for li, s in scores.items():
            k = _KEYS.get(li)
            if k is None or k.shape[2] != n:
                continue
            kraw = k[0].float()                                        # [n_kv, n, d]
            kf = torch.nn.functional.normalize(kraw, dim=-1)
            # Centred view: transformer key spaces are strongly anisotropic (a
            # large shared mean direction), so raw cosine sits near 0.66 for
            # ANY pair and discriminates poorly. Removing the per-head mean
            # direction is what makes a similarity threshold meaningful, and
            # tells us whether the geometry is usable at all.
            kc = torch.nn.functional.normalize(kraw - kraw.mean(1, keepdim=True), dim=-1)
            top = s[0].topk(args.budget, dim=-1).indices               # [n_kv, B]
            rnd = torch.stack([torch.randperm(n, generator=g)[:args.budget]
                               for _ in range(kf.shape[0])]).to(kf.device)

            for name, idx, basis in (("top", top, kf), ("rand", rnd, kf),
                                     ("ctop", top, kc), ("crand", rnd, kc)):
                sub = torch.gather(basis, 1, idx.unsqueeze(-1).expand(-1, -1, basis.shape[-1]))
                sim = sub @ sub.transpose(1, 2)                        # [n_kv, B, B]
                off = ~torch.eye(args.budget, dtype=torch.bool, device=sim.device)
                per_layer[f"sim_{name}"][li] += sim[:, off].mean().item()
                per_layer[f"er_{name}"][li] += sum(
                    eff_rank(sim[h]) for h in range(sim.shape[0])) / sim.shape[0]
                per_layer[f"clu_{name}"][li] += sum(
                    n_clusters(sim[h], args.tau) for h in range(sim.shape[0])) / sim.shape[0]

            sim_all = kf @ kf.transpose(1, 2)
            off_all = ~torch.eye(n, dtype=torch.bool, device=sim_all.device)
            per_layer["sim_all"][li] += sim_all[:, off_all].mean().item()
            counts[li] += 1

    counts = counts.clamp_min(1)
    for k in per_layer:
        per_layer[k] /= counts

    B = args.budget
    print(f"R2, {len(prompts)} rag_dev prompts, budget B={B}, cosine threshold {args.tau}")
    print(f"mean over 8 KV heads. eff_rank/B and clusters/B are the headroom the")
    print(f"geometric idea claims; 1.0 means no redundancy to exploit.\n")
    print(f"{'layer':>6}{'sim(top)':>10}{'sim(rand)':>11}{'sim(all)':>10}"
          f"{'effR(top)':>11}{'effR/B':>8}{'clu(top)':>10}{'clu/B':>7}")
    for li in range(model.config.num_hidden_layers):
        print(f"{li:>6}{per_layer['sim_top'][li]:>10.3f}{per_layer['sim_rand'][li]:>11.3f}"
              f"{per_layer['sim_all'][li]:>10.3f}{per_layer['er_top'][li]:>11.1f}"
              f"{per_layer['er_top'][li]/B:>8.2f}{per_layer['clu_top'][li]:>10.1f}"
              f"{per_layer['clu_top'][li]/B:>7.2f}")
    m = {k: float(v.mean()) for k, v in per_layer.items()}
    print(f"{'MEAN':>6}{m['sim_top']:>10.3f}{m['sim_rand']:>11.3f}{m['sim_all']:>10.3f}"
          f"{m['er_top']:>11.1f}{m['er_top']/B:>8.2f}{m['clu_top']:>10.1f}"
          f"{m['clu_top']/B:>7.2f}")

    print()
    print("reading -- the RANDOM subset is the control, and the CENTRED space is")
    print("the one that means anything:")
    print(f"  raw cosine over ALL keys is {m['sim_all']:.3f}: the key space is strongly")
    print( "  anisotropic, so any two keys look similar and a raw cosine threshold")
    print( "  discriminates almost nothing. Centring removes the shared mean direction")
    print(f"  (random-subset mean cosine after centring: {m['sim_crand']:+.3f}).")
    print()
    print(f"  {'metric':<24}{'top-k':>9}{'random':>9}{'ratio':>8}")
    for lbl, a, b in (("eff. rank (raw)", "er_top", "er_rand"),
                      ("eff. rank (centred)", "er_ctop", "er_crand"),
                      ("clusters (raw)", "clu_top", "clu_rand"),
                      ("clusters (centred)", "clu_ctop", "clu_crand")):
        print(f"  {lbl:<24}{m[a]:>9.2f}{m[b]:>9.2f}{m[a]/max(m[b],1e-9):>8.2f}")
    excess = 1.0 - m["clu_ctop"] / max(m["clu_crand"], 1e-9)
    print()
    if excess < 0.05:
        print(f"  VERDICT: in centred space top-k keeps {m['clu_ctop']:.1f} of {B} slots")
        print(f"  distinct, against {m['clu_crand']:.1f} for RANDOM selection -- only")
        print(f"  {excess:.0%} worse. Top-k is already about as diverse as chance allows,")
        print( "  so a diversity-aware policy has essentially nothing to recover here.")
        print(f"  The effective-rank gap ({m['er_ctop']/max(m['er_crand'],1e-9):.2f}) is real but is a")
        print( "  soft concentration effect, not the duplicate-slot waste the geometric")
        print( "  proposal targets.")
    else:
        print(f"  VERDICT: headroom -- top-k wastes {excess:.0%} of the budget on")
        print( "  near-duplicates relative to random selection, in centred space.")
    print()
    print( "  CAVEAT: measured at 41-330 token prompts, which contain almost no filler.")
    print( "  Long contexts with repeated boilerplate could be far more redundant, and")
    print( "  that regime is unreachable in this repository.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": MODEL, "cases": len(prompts), "budget": B, "tau": args.tau,
        "mean": m,
        "per_layer": {k: [round(x, 5) for x in v.tolist()] for k, v in per_layer.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
