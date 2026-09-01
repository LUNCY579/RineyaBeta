"""Stage C1c: how much of DAST's state signal actually comes from DEPTH?

`L_state` is only meaningful if the thing it measures needs the depth stack. These
controls bound that. Each fits the SAME shared linear readout used by C1b -- the
estimator analogue of `SharedStateProjector`, which V1 also shares across depth --
and differs only in which hidden states it is given.

    real            layers 7/14/21/27, the V1 checkpoints.
    embedding_only  layer 0 (token embeddings, no decoder layer) repeated at all
                    four depths. A shared readout sees IDENTICAL input at every
                    depth, so whatever it scores is what survives with zero depth
                    information. This is the floor for "depth contributes".
    embedding_x4    layers 0/1/2/3: four genuinely different but almost entirely
                    pre-computation states. If this matches `real`, the 28-layer
                    stack contributes nothing that 3 layers do not.
    shuffled_pairing
                    real layers, but each problem is scored against ANOTHER
                    problem's targets. Destroys the problem-target correspondence
                    while preserving both marginal geometries. This is the chance
                    floor.

Read the output alongside the trained runs: `state_ordered` reached L_state 0.2629
with the projector it trained for 250 steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.config import load_config  # noqa: E402
from dast.depth_cache import DepthCache  # noqa: E402
from dast.depth_probe import permutation_landscape  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_landscape import build_targets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="an --all-layers --first-layer 0 cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--real-layers", default="7,14,21,27")
    args = parser.parse_args()

    config = load_config(args.config)
    cache = DepthCache.load(args.cache)
    position = {layer: index for index, layer in enumerate(cache.layers)}
    real = [int(part) for part in args.real_layers.split(",")]
    for layer in [*real, 0, 1, 2, 3]:
        if layer not in position:
            raise SystemExit(f"cache lacks layer {layer}; rebuild with --all-layers --first-layer 0")

    targets, _ = build_targets(config, cache, args.stages)
    print(f"targets {tuple(targets.shape)} centred within problem", flush=True)

    conditions = {
        "real": cache.hidden[:, [position[layer] for layer in real], :],
        "embedding_only": cache.hidden[:, [position[0]] * args.stages, :],
        "embedding_x4": cache.hidden[:, [position[index] for index in range(args.stages)], :],
    }
    # Chance floor: rotate the problem index so hidden states meet the wrong targets.
    rotated = torch.roll(torch.arange(cache.num_problems), shifts=1)
    conditions["shuffled_pairing"] = cache.hidden[:, [position[layer] for layer in real], :][rotated]

    results = {}
    for name, hidden in conditions.items():
        landscape = permutation_landscape(
            hidden.contiguous(), targets, splits=args.splits, seed=1701
        )
        identity = next(
            row for row in landscape["permutations_best_first"]
            if tuple(row["perm"]) == tuple(range(args.stages))
        )
        results[name] = {
            "identity_L_state": identity["L_state_mean"],
            "identity_order_margin": identity["order_margin_mean"],
            "identity_top1": identity["top1_mean"],
            "identity_rank_median": identity["rank_median"],
            "best_perm": landscape["summary"]["best_perm"],
            "best_L_state": landscape["permutations_best_first"][0]["L_state_mean"],
            "L_state_spread": landscape["summary"]["L_state_spread_mean"],
            "spread_fraction_of_L_state": landscape["summary"][
                "L_state_spread_as_fraction_of_L_state"
            ],
        }
        print(
            f"  {name:20s} L_state {results[name]['identity_L_state']:.5f}  "
            f"margin {results[name]['identity_order_margin']:+.4f}  "
            f"top1 {results[name]['identity_top1']:.4f}  "
            f"spread {results[name]['spread_fraction_of_L_state']:.2%}",
            flush=True,
        )

    print("\n=== C1c: where the state signal lives ===")
    print(f"  {'condition':20s} {'L_state':>9} {'margin':>9} {'top1':>7} {'spread':>8}")
    for name, row in results.items():
        print(f"  {name:20s} {row['identity_L_state']:>9.5f} {row['identity_order_margin']:>+9.4f} "
              f"{row['identity_top1']:>7.4f} {row['spread_fraction_of_L_state']:>7.2%}")
    print(f"  {'TRAINED state_ordered':20s} {0.2629:>9.5f} {0.9350:>+9.4f} {0.8332:>7.4f}"
          "        (250 steps, DEV)")

    payload = {
        "num_problems": cache.num_problems,
        "num_stages": args.stages,
        "splits": args.splits,
        "real_layers": real,
        "conditions": results,
        "trained_state_ordered_reference": {
            "L_state": 0.2629, "order_margin": 0.9350, "top1": 0.8332,
        },
        "cache_metadata": cache.metadata,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
