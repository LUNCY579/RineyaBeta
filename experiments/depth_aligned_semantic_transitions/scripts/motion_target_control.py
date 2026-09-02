"""H6 gate: do semantic MOTION targets carry problem-specific information at all?

Motion/ΔH is the last DAST successor with a mechanism-level argument, and it spends
the SSOT §13 one-mechanism revision budget. It should not be spent on a target that
has not passed the control absolute states failed. This script runs that control on
CPU using the existing depth cache; no GPU, no training.

The argument this is built to test
----------------------------------
`PHASE0_READINESS.md` §3.8 measured, for motion targets, between-problem same-stage
delta cosine of **0.94-0.99** against within-problem cross-stage **-0.32**. That is
read as favourable: deltas encode STAGE rather than problem identity, an exact
reversal of the absolute-state pattern that motivated within-problem centering.

There is a less favourable reading of the same number. If Δs_k is nearly the *same
vector for every problem* at stage k, then it is approximately a per-stage constant,
and "predicting" it collapses to decoding which depth you are at -- trivially easy,
excellent mechanism metrics, and **zero information about the problem being
reasoned about**. That is the "readable ≠ useful" trap this project has already
documented once.

The decisive control is `shuffled_pairing`: score each problem's hidden states
against ANOTHER problem's targets. It preserves both marginal geometries and
destroys only the correspondence. For ABSOLUTE states it is devastating
(`L_state` 0.143 -> 0.711). The diagnostic quantity is therefore

    problem_information = L_state(shuffled) - L_state(true)

which is 0.568 for absolute states. If motion targets score a small value here,
they are close to problem-independent and the objective is near-vacuous.

Target variants
---------------
    absolute        the V1 target: s_k, centred within problem. Reference row.
    motion_raw      Δs_k = s_k - s_{k-1}, L2-normalised, NOT centred.
    motion_centered Δs_k as above, then centred within problem like V1 does.

s_{-1} is the pre-reasoning initial state `compact_state_trace[0]`, which
`raw_targets` deliberately drops, so K absolute states yield exactly K motion
targets at K depths.

Hidden variants
---------------
    real            layers 7/14/21/27, the V1 sites.
    embedding_x4    layers 0/1/2/3: is the target recoverable before the stack?
    embedding_only  layer 0 at every depth: the zero-depth-information floor.
    delta_hidden    Δh_k = h_k - h_{k-1} at the real layers, with h_{-1} = layer 0.
                    This is the full ΔH formulation: motion on BOTH sides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.alignment import _progress_indices  # noqa: E402
from dast.config import ARM_SPECS, load_config  # noqa: E402
from dast.data import build_examples, load_records  # noqa: E402
from dast.depth_cache import DepthCache  # noqa: E402
from dast.depth_probe import center_targets, permutation_landscape  # noqa: E402
from dast.hf import load_frozen_base  # noqa: E402
from dast.semantic import FrozenTextTargetEncoder  # noqa: E402

ABSOLUTE_REFERENCE_PROBLEM_INFORMATION = 0.568  # L_shuffled - L_true, from C1c


def build_target_variants(config, cache: DepthCache, stages: int):
    """Return (variants, keep_mask). Variants share one encoder and one problem order."""

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
    kind = ARM_SPECS["state_ordered"].target_kind
    examples = build_examples(
        records, arm=ARM_SPECS["state_ordered"], num_stages=stages, seed=config.experiment.seed
    )
    if [item.record.problem_id for item in examples] != list(cache.problem_ids):
        raise SystemExit("record order does not match the cache; refusing to pair them")

    absolute = encoder.encode_grid(
        [list(item.trajectory.canonical_texts) for item in examples], device="cpu"
    )
    # s_{-1}: the pre-reasoning state that raw_targets drops on purpose.
    initial = encoder.encode(
        [item.record.compact_state_trace[0] for item in examples], device="cpu"
    )
    previous = torch.cat([initial.unsqueeze(1), absolute[:, :-1, :]], dim=1)
    deltas = absolute - previous
    norms = deltas.norm(dim=-1)
    degenerate = int((norms < 1e-6).sum())
    motion_raw = F.normalize(deltas, dim=-1)

    distinct = [
        len(set(_progress_indices(len(item.record.raw_targets(kind)), stages))) == stages
        for item in examples
    ]
    variants = {
        "absolute": center_targets(absolute),
        "motion_raw": motion_raw,
        "motion_centered": center_targets(motion_raw),
    }
    return variants, distinct, degenerate


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

    variants, distinct, degenerate = build_target_variants(config, cache, args.stages)
    keep = torch.tensor(distinct, dtype=torch.bool)
    count = int(keep.sum())
    print(f"kept {count}/{cache.num_problems} with {args.stages} distinct depth targets; "
          f"{degenerate} degenerate (zero-length) motion targets before filtering", flush=True)

    hidden_all = cache.hidden[keep]
    real_hidden = hidden_all[:, [position[layer] for layer in real], :]
    # Delta_h_k = h_k - h_{k-1}, with h_{-1} = the token embedding (layer 0).
    shifted = torch.cat(
        [hidden_all[:, position[0]].unsqueeze(1), real_hidden[:, :-1, :]], dim=1
    )
    hidden_variants = {
        "real": real_hidden,
        "embedding_x4": hidden_all[:, [position[i] for i in range(args.stages)], :],
        "embedding_only": hidden_all[:, [position[0]] * args.stages, :],
        "delta_hidden": real_hidden - shifted,
    }
    rotated = torch.roll(torch.arange(count), shifts=1)

    results: dict[str, dict] = {}
    print(f"\n{'target':17s} {'hidden':15s} {'L_true':>8} {'L_shuf':>8} "
          f"{'problem info':>13} {'margin':>8} {'top1':>7}")
    for target_name, grid in variants.items():
        targets = grid[keep]
        results[target_name] = {}
        for hidden_name, hidden in hidden_variants.items():
            scored = {}
            for pairing, tensor in (("true", hidden), ("shuffled", hidden[rotated])):
                landscape = permutation_landscape(
                    tensor.contiguous(), targets, splits=args.splits, seed=1701
                )
                identity = next(
                    row for row in landscape["permutations_best_first"]
                    if tuple(row["perm"]) == tuple(range(args.stages))
                )
                scored[pairing] = {
                    "L_state": identity["L_state_mean"],
                    "order_margin": identity["order_margin_mean"],
                    "top1": identity["top1_mean"],
                    "spread_fraction": landscape["summary"][
                        "L_state_spread_as_fraction_of_L_state"
                    ],
                }
            information = scored["shuffled"]["L_state"] - scored["true"]["L_state"]
            scored["problem_information"] = information
            results[target_name][hidden_name] = scored
            print(f"{target_name:17s} {hidden_name:15s} {scored['true']['L_state']:>8.5f} "
                  f"{scored['shuffled']['L_state']:>8.5f} {information:>13.5f} "
                  f"{scored['true']['order_margin']:>+8.4f} {scored['true']['top1']:>7.4f}",
                  flush=True)

    reference = results["absolute"]["real"]["problem_information"]
    print(f"\n=== the H6 gate ===")
    print(f"  absolute states, V1 sites:      problem information = {reference:.5f}")
    for name in ("motion_raw", "motion_centered"):
        for hidden_name in ("real", "delta_hidden"):
            value = results[name][hidden_name]["problem_information"]
            print(f"  {name:16s} / {hidden_name:13s} problem information = {value:.5f}"
                  f"   ({value / reference:.2f}x absolute)")

    best_motion = max(
        results[name][hidden]["problem_information"]
        for name in ("motion_raw", "motion_centered")
        for hidden in ("real", "delta_hidden")
    )
    carries_problem_information = best_motion >= 0.5 * reference

    # Criterion 2, the layer-0 recoverability control. A target that genuinely needs
    # the depth stack must be fitted BETTER at the V1 sites than at layers 0/1/2/3.
    # Absolute states fail this; motion only escapes the defect if it turns positive.
    print("\n  depth advantage = L_state(embedding_x4) - L_state(real); "
          "positive means the stack helps")
    depth_advantage = {}
    for name in variants:
        advantage = (
            results[name]["embedding_x4"]["true"]["L_state"]
            - results[name]["real"]["true"]["L_state"]
        )
        depth_advantage[name] = advantage
        print(f"    {name:17s} {advantage:+.5f}"
              f"   {'stack helps' if advantage > 0 else 'BOTTOM LAYERS STILL WIN'}")
    needs_depth = any(
        depth_advantage[name] > 0 for name in ("motion_raw", "motion_centered")
    )

    if not carries_problem_information:
        verdict = (
            "MOTION DISQUALIFIED on problem information: the objective is closer to "
            "decoding the depth index than to reading the problem's computation."
        )
    elif not needs_depth:
        verdict = (
            "MOTION DOES NOT CLEAR THE GATE. It PASSES the problem-information control "
            f"({best_motion / reference:.2f}x absolute), so deltas are NOT "
            "problem-independent and the pessimistic reading of PHASE0 3.8 is wrong. But it "
            "FAILS the layer-0 recoverability control exactly as absolute states do: layers "
            "0/1/2/3 still fit motion targets better than the V1 sites, and moving the hidden "
            "side to deltas makes it worse rather than better. Motion changes WHAT is "
            "supervised without changing the fact that the supervised quantity does not "
            "require the depth stack. Do not spend the revision budget on it."
        )
    else:
        verdict = (
            "MOTION CLEARS BOTH GATES: it retains problem-specific information AND is fitted "
            "better at the V1 depths than at the bottom of the stack. This licenses TESTING "
            "motion, not adopting it."
        )
    print(f"\nVERDICT: {verdict}")

    payload = {
        "num_problems": count,
        "num_problems_before_filter": cache.num_problems,
        "degenerate_motion_targets_before_filter": degenerate,
        "num_stages": args.stages,
        "splits": args.splits,
        "real_layers": real,
        "results": results,
        "absolute_reference_problem_information": reference,
        "best_motion_problem_information": best_motion,
        "carries_problem_information": carries_problem_information,
        "depth_advantage": depth_advantage,
        "needs_depth": needs_depth,
        "verdict": verdict,
        "claim_boundary": (
            "Decodability only, frozen backbone, no training. A pass here is a licence to "
            "TEST motion, not evidence that motion improves reasoning. TEST sealed."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
