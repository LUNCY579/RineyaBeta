"""Backfill a B2-T run summary whose trajectory completed but whose summary crashed.

A stale `result.global_steps` reference raised AttributeError AFTER the trajectory was
fully streamed and flushed, so several EARLY-only runs have complete, valid 120-row
trajectories and no `grad_trajectory_summary.json`. The analyser reads the init digest
from that summary to enforce the matched-initialisation contract, so without it those
seeds are correctly rejected as unusable.

Re-running the arms would cost ~5 minutes each. Instead this rebuilds the model
exactly as the runner does -- same seed, same construction order, no training -- and
recomputes the digest, which is a deterministic function of (seed, config). Roughly 30
seconds per arm.

Guard: refuses if a summary already exists, and refuses if the trajectory is absent or
short of the requested window, so it cannot manufacture a summary for data that is not
there.

Usage
-----
    python scripts/b2t_backfill_summary.py --config configs/diag_b2traj_s101_state_ordered.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.config import load_config  # noqa: E402
from dast.hf import build_hf_components  # noqa: E402
from dast.model import DASTModel  # noqa: E402
from dast.training import set_global_seed  # noqa: E402


def init_digest(config) -> tuple[str, int]:
    """Rebuild exactly as the runner does and hash the initial trainable state."""

    set_global_seed(
        config.experiment.seed,
        deterministic_algorithms=config.training.deterministic_algorithms,
    )
    base, _tokenizer, _info = build_hf_components(config, for_training=True)
    model = DASTModel(
        base,
        checkpoint_fractions=config.model.checkpoint_fractions,
        state_dim=config.semantic.state_dim,
        projector_hidden_dim=config.semantic.projector_hidden_dim,
        pooling=config.model.pooling,
        feedback_enabled=config.model.feedback_enabled,
        feedback_scale=config.model.feedback_scale,
        checkpoint_readout=config.model.checkpoint_readout,
    )
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
    digest = hashlib.sha256(
        b"".join(
            name.encode() + state[name].detach().cpu().float().numpy().tobytes()
            for name in sorted(state)
        )
    ).hexdigest()
    return digest, int(torch.initial_seed())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expect-steps", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.resolved_path(config.experiment.output_dir)
    trajectory = output_dir / "grad_trajectory.jsonl"
    summary_path = output_dir / "grad_trajectory_summary.json"

    if not trajectory.exists():
        raise SystemExit(f"no trajectory at {trajectory}; refusing to invent a summary")
    if summary_path.exists() and not args.force:
        print(f"summary already present, leaving alone: {summary_path}")
        return 0

    rows = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines()]
    measured = [row for row in rows if row.get("measured")]
    if not measured:
        raise SystemExit("trajectory contains no measured rows; refusing")
    highest = max(row["global_step"] for row in measured)
    if highest < args.expect_steps:
        raise SystemExit(
            f"trajectory reaches global step {highest}, short of the requested "
            f"{args.expect_steps}; refusing to summarise a truncated window"
        )

    digest, torch_seed = init_digest(config)
    summary = {
        "arm": config.experiment.arm,
        "seed": config.experiment.seed,
        "config_sha256": config.canonical_sha256(),
        "config_path": str(args.config),
        "global_steps": highest,
        "optimizer_steps": max(row["optimizer_step"] for row in measured),
        "measured_steps": len(measured),
        "stopped_early": True,
        "stop_after_step": args.expect_steps,
        "torch_initial_seed": torch_seed,
        "init_trainable_state_sha256": digest,
        "trajectory": str(trajectory),
        "backfilled": True,
        "backfill_note": (
            "The run completed its trajectory and then crashed writing this summary "
            "(stale result.global_steps reference after EarlyStop). The digest is "
            "recomputed by rebuilding the model with the same seed and construction "
            "order; it is a deterministic function of (seed, config), not of the run."
        ),
        "claim_boundary": (
            "B2-T diagnosis only. No capability endpoint; generation disabled. DEV only, "
            "TEST sealed. Accuracy from this run must not be reported."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{config.experiment.arm}@{config.experiment.seed}: {len(measured)} measured "
        f"steps to {highest}, digest {digest[:16]}\nwrote {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
