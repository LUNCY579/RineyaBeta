"""Stage C1: how much does the FROZEN model care which semantic state sits where?

Replaces and generalises `scripts/permutation_preference.py`, whose input cache no
committed script produced. Two measurements, both training-free, both CPU:

C1a  ``--mode profile``     every usable layer x every semantic stage. Asks whether
                            depth has ANY chronological gradient, without assuming
                            K checkpoints or an assignment. A chronological depth
                            axis predicts the argmax layer to rise with stage.
C1b  ``--mode permutation`` exhaustive state-to-depth assignment ranking at K
                            fixed depths, over many random splits.

Why K=8 is NOT offered
----------------------
The plan that motivated this script proposed K=8. On this corpus that is not
measurable. ``align_trajectory`` subsamples each problem's ``compact_state_trace``
via ``_progress_indices``, and the trace has 3-10 entries with a mode of 4. The
count of DEV problems receiving K *distinct* depth targets is:

    K=4  2303/2400 (96.0%)      K=6   604/2400 (25.2%)
    K=5  1379/2400 (57.5%)      K=8   110/2400 ( 4.6%)

At K=8 the modal problem gets four distinct targets each repeated twice, so any
permutation exchanging two positions holding the same target is a literal no-op and
the landscape is flattened by the *target construction*, not by the network. That is
the artifact the study exists to rule out. ``--require-distinct`` therefore drops
problems whose depth targets are not all distinct, and K is capped at 6 where the
surviving n still supports a ridge fit.

Usage
-----
    python scripts/permutation_landscape.py --mode permutation \\
        --cache artifacts/depth_cache_k4_dev2400.pt \\
        --config configs/g3_s29_state_ordered.yaml --out artifacts/c1b_k4.json

    python scripts/permutation_landscape.py --mode profile \\
        --cache artifacts/depth_cache_all27_dev2400.pt \\
        --config configs/g3_s29_state_ordered.yaml --stages 4 --out artifacts/c1a_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.alignment import _progress_indices  # noqa: E402
from dast.config import ARM_SPECS, load_config  # noqa: E402
from dast.data import build_examples, load_records  # noqa: E402
from dast.depth_cache import DepthCache  # noqa: E402
from dast.depth_probe import center_targets, layer_stage_profile, permutation_landscape  # noqa: E402
from dast.hf import load_frozen_base  # noqa: E402
from dast.semantic import FrozenTextTargetEncoder  # noqa: E402

MAX_SUPPORTED_STAGES = 6


def build_targets(config, cache: DepthCache, stages: int) -> tuple[torch.Tensor, list[bool]]:
    """Encode centred canonical targets at `stages` resolution for the cache's problems.

    Returns the target grid and a per-problem mask marking whether that problem's
    depth targets are all DISTINCT. Only the frozen input embedding is loaded, so
    this is the same target geometry the runs used.
    """

    model, tokenizer = load_frozen_base(config)
    encoder = FrozenTextTargetEncoder(
        model.get_input_embeddings(),
        tokenizer,
        state_dim=config.semantic.state_dim,
        projection_seed=config.semantic.target_projection_seed,
        max_tokens=config.semantic.target_max_tokens,
    )
    records = load_records(
        config.data_path_for_split(cache.metadata["split"]),
        split=cache.metadata["split"],
        limit=cache.num_problems,
    )
    examples = build_examples(
        records, arm=ARM_SPECS["state_ordered"], num_stages=stages, seed=config.experiment.seed
    )
    ids = [item.record.problem_id for item in examples]
    if tuple(ids) != cache.problem_ids:
        raise SystemExit(
            "cache problem ids do not match the freshly loaded records; refusing to pair them"
        )
    grid = encoder.encode_grid(
        [list(item.trajectory.canonical_texts) for item in examples], device="cpu"
    )
    distinct = [
        len(set(_progress_indices(length, stages))) == stages
        for length in cache.trajectory_lengths
    ]
    return center_targets(grid), distinct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("permutation", "profile"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stages", type=int, default=None, help="K; defaults to the cache's")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--require-distinct",
        action="store_true",
        default=True,
        help="drop problems whose K depth targets are not all distinct (default on)",
    )
    parser.add_argument("--allow-duplicate-targets", dest="require_distinct", action="store_false")
    args = parser.parse_args()

    config = load_config(args.config)
    cache = DepthCache.load(args.cache)
    stages = args.stages or cache.num_stages
    if stages > MAX_SUPPORTED_STAGES:
        raise SystemExit(
            f"K={stages} exceeds the supported {MAX_SUPPORTED_STAGES}: see this script's docstring. "
            "Only 4.6% of DEV problems have 8 distinct depth targets, so a K=8 landscape measures "
            "target duplication rather than depth structure."
        )
    print(f"cache: hidden {tuple(cache.hidden.shape)} at layers {cache.layers}", flush=True)

    targets, distinct = build_targets(config, cache, stages)
    keep = torch.tensor(
        [True] * cache.num_problems if not args.require_distinct else distinct, dtype=torch.bool
    )
    dropped = int((~keep).sum())
    print(
        f"targets: {tuple(targets.shape)} centred within problem; "
        f"K={stages}; kept {int(keep.sum())}/{cache.num_problems} "
        f"(dropped {dropped} with duplicated depth targets)",
        flush=True,
    )
    if int(keep.sum()) < 100:
        raise SystemExit(f"only {int(keep.sum())} problems survive the distinctness filter")

    hidden = cache.hidden[keep]
    targets = targets[keep]

    if args.mode == "permutation":
        if hidden.shape[1] != stages:
            raise SystemExit(
                f"cache has {hidden.shape[1]} checkpoints but K={stages}; build a matching cache"
            )
        result = permutation_landscape(
            hidden, targets, splits=args.splits, ridge=args.ridge, seed=args.seed
        )
        result["dropped_duplicate_target_problems"] = dropped
        result["layers"] = list(cache.layers)
        summary = result["summary"]
        print("\n=== C1b: state-to-depth assignment landscape ===")
        print(f"  n={result['num_problems']}  K={stages}  perms={result['num_permutations']}"
              f"  exhaustive={result['exhaustive']}  splits={result['splits']}")
        print(f"  {'rank':>4} {'perm':>18} {'L_state':>9} {'margin':>9} {'top1':>7} {'tau':>6} "
              f"{'med.rank':>9} {'#1st':>5}  name")
        for index, row in enumerate(result["permutations_best_first"], start=1):
            print(f"  {index:>4} {str(tuple(row['perm'])):>18} {row['L_state_mean']:>9.5f} "
                  f"{row['order_margin_mean']:>9.4f} {row['top1_mean']:>7.4f} "
                  f"{row['kendall_tau_vs_identity']:>+6.2f} {row['rank_median']:>9} "
                  f"{row['times_ranked_first']:>5}  {row['name']}")
        print(f"\n  identity: median rank {summary['identity_rank_median']}, "
              f"first in {summary['identity_times_first']}/{args.splits} splits, "
              f"L_state {summary['identity_L_state_mean']:.5f}")
        print(f"  reverse : median rank {summary['reverse_rank_median']}, "
              f"first in {summary['reverse_times_first']}/{args.splits} splits, "
              f"L_state {summary['reverse_L_state_mean']:.5f}")
        print(f"  L_state spread across all permutations: {summary['L_state_spread_mean']:.5f} "
              f"= {summary['L_state_spread_as_fraction_of_L_state']:.2%} of L_state itself")
    else:
        result = layer_stage_profile(
            hidden, targets, cache.layers, splits=args.splits, ridge=args.ridge, seed=args.seed
        )
        result["dropped_duplicate_target_problems"] = dropped
        print("\n=== C1a: layer x stage decodability (held-out cosine) ===")
        header = "  layer " + "".join(f"  stage{stage}" for stage in range(stages))
        print(header + "    argmax")
        for position, layer in enumerate(cache.layers):
            row = result["cosine_mean"][position]
            best = max(range(stages), key=lambda stage: row[stage])
            print(f"  {layer:>5} " + "".join(f"  {value:7.4f}" for value in row) + f"    stage{best}")
        print(f"\n  best layer per stage: {result['best_layer_per_stage']}")
        print(f"  monotone in stage   : {result['monotone_in_stage']}")
        print(f"  spearman(stage, best layer) = {result['spearman_stage_vs_best_layer']:+.4f}")

    result["cache_metadata"] = cache.metadata
    result["require_distinct"] = args.require_distinct
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
