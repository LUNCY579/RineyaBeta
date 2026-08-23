#!/usr/bin/env python3
"""Measure low-rank structure in every weight matrix. Pure measurement, no training.

Factorizing `W (d x k)` into `U (d x r) V (r x k)` is only worth doing where the
spectrum is actually concentrated: the replacement costs `r*(d+k)` parameters
against `d*k`, so it saves nothing unless `r < d*k/(d+k)`. For Qwen3-0.6B's shapes
that break-even rank is 683 for a 1024x2048 projection and 768 for a 1024x3072 MLP
matrix -- both well below full rank, so the question is entirely empirical.

This answers, per matrix and without touching the GPU: how much energy sits in how
few singular values, and what would factorization actually save at that rank?

Reported per projection type and per depth band, because the interesting result
would be *structure* -- if effective rank varies systematically with depth or with
projection role, a rank schedule beats a uniform rank at equal size, and that is a
design rule rather than a knob.

Deliberately does not recommend a rank. Nothing here measures capability; a
spectrum says what is cheap to remove, not what is safe to remove. That needs the
gates.

Usage: python scripts/measure/weight_spectra.py [layer_stride] [model_dir]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
ENERGY_LEVELS = (0.90, 0.95, 0.99)
PROJ_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def effective_rank(sv: torch.Tensor, energy: float) -> int:
    """Smallest r whose top-r singular values hold `energy` of the squared total."""
    power = sv.double() ** 2
    cumulative = torch.cumsum(power, 0) / power.sum()
    return int(torch.searchsorted(cumulative, energy).item()) + 1


def main() -> None:
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    model_dir = ROOT / (sys.argv[2] if len(sys.argv) > 2
                        else "artifacts/rineya-core-en32k-28L")
    torch.set_num_threads(4)  # be polite; other jobs may be running

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = config["num_hidden_layers"]
    sampled = list(range(0, layers, stride))
    print(f"model  : {model_dir.name}  ({layers} layers, sampling every {stride} "
          f"-> {len(sampled)} layers)")
    print(f"energy : {', '.join(f'{e:.0%}' for e in ENERGY_LEVELS)}\n")

    rows = []
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as handle:
        for layer in sampled:
            for proj in PROJ_ORDER:
                key = f"model.layers.{layer}.self_attn.{proj}.weight"
                if proj.startswith(("gate", "up", "down")):
                    key = f"model.layers.{layer}.mlp.{proj}.weight"
                if key not in handle.keys():
                    continue
                w = handle.get_tensor(key).float()
                d, k = w.shape
                sv = torch.linalg.svdvals(w)
                full = d * k
                breakeven = full / (d + k)
                entry = {"layer": layer, "proj": proj, "shape": [d, k],
                         "full_params": full, "breakeven_rank": round(breakeven, 1)}
                for e in ENERGY_LEVELS:
                    r = effective_rank(sv, e)
                    factored = r * (d + k)
                    entry[f"rank_{int(e*100)}"] = r
                    entry[f"saving_{int(e*100)}"] = full - factored
                rows.append(entry)
            print(f"  layer {layer:>2d} done", flush=True)

    print(f"\n=== by projection (mean over {len(sampled)} sampled layers) ===")
    print(f"{'proj':11s} {'shape':>12s} {'break-even':>10s} "
          + "".join(f"{f'r@{int(e*100)}%':>9s}{'save':>9s}" for e in ENERGY_LEVELS))
    by_proj = defaultdict(list)
    for r in rows:
        by_proj[r["proj"]].append(r)
    for proj in PROJ_ORDER:
        v = by_proj.get(proj)
        if not v:
            continue
        d, k = v[0]["shape"]
        line = f"{proj:11s} {f'{d}x{k}':>12s} {v[0]['breakeven_rank']:>10.0f} "
        for e in ENERGY_LEVELS:
            rk = sum(x[f"rank_{int(e*100)}"] for x in v) / len(v)
            sv_ = sum(x[f"saving_{int(e*100)}"] for x in v) / len(v)
            line += f"{rk:>9.0f}{sv_/1e6:>8.2f}M"
        print(line)

    print(f"\n=== by depth band, rank at 99% energy (is there structure?) ===")
    bands = {"early 0-6": range(0, 7), "mid 7-20": range(7, 21), "late 21-27": range(21, 28)}
    print(f"{'band':12s} " + "".join(f"{p.split('_')[0]:>9s}" for p in PROJ_ORDER))
    for name, rng in bands.items():
        line = f"{name:12s} "
        for proj in PROJ_ORDER:
            v = [x for x in rows if x["proj"] == proj and x["layer"] in rng]
            line += f"{(sum(x['rank_99'] for x in v)/len(v)) if v else float('nan'):>9.0f}"
        print(line)

    print(f"\n=== whole-model extrapolation (sampled mean x {layers} layers) ===")
    resident = 596_049_920 if config["vocab_size"] > 100_000 else 474_021_888
    for e in ENERGY_LEVELS:
        per_layer = sum(sum(x[f"saving_{int(e*100)}"] for x in by_proj[p]) / len(by_proj[p])
                        for p in PROJ_ORDER if by_proj.get(p))
        total = per_layer * layers
        print(f"  at {int(e*100)}% energy: save {total/1e6:>7.2f}M params "
              f"-> {(resident-total)/1e6:.1f}M total ({100*(resident-total)/resident-100:+.2f}%)"
              f"{'   (negative = factorization costs more than it saves)' if total < 0 else ''}")

    out = ROOT / "manifests" / f"weight_spectra_{model_dir.name}.json"
    out.write_text(json.dumps({"model": model_dir.name, "layer_stride": stride,
                               "sampled_layers": sampled,
                               "energy_levels": list(ENERGY_LEVELS),
                               "per_matrix": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\nNote: a spectrum says what is cheap to remove, not what is safe to "
          "remove. Capability needs the gates.")


if __name__ == "__main__":
    main()
