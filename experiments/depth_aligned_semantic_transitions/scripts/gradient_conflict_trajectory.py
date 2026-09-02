"""B2-T: run one instrumented training arm and stream the gradient-conflict trajectory.

FROZEN protocol: docs/PREREGISTRATION_B2_TRAJECTORY.md. Read it before changing
anything here; the schedule, the windows and the decision rule are preregistered.

This trains a real arm with a read-only probe attached. The probe takes
`torch.autograd.grad` on `L_LM` and `L_state` separately at scheduled steps, before
the production backward, and cannot move the optimizer -- pinned by a bit-identical
test in `tests/test_grad_conflict.py`.

Rows are streamed to JSONL as they are measured, on purpose. Unlike a partial
accuracy run, which is worthless, a partial gradient trajectory is a valid
measurement of the steps it covers. If the accelerator is lost mid-run the data
already written stands.

Usage
-----
    python scripts/gradient_conflict_trajectory.py \\
        --config configs/diag_b2traj_s29_state_ordered.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dast.config import load_config  # noqa: E402
from dast.grad_conflict import TrajectoryProbe  # noqa: E402
from dast.hf import build_hf_components  # noqa: E402
from dast.model import DASTModel  # noqa: E402
from dast.training import train  # noqa: E402
from gradient_conflict import trainer_running  # noqa: E402
from teacher_forced_endpoint import accelerator_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="JSONL trajectory; default is in the run dir")
    parser.add_argument("--min-free-mib", type=int, default=4000)
    parser.add_argument("--max-busy-util", type=int, default=60)
    parser.add_argument("--dense-until", type=int, default=120)
    parser.add_argument("--mid-until", type=int, default=400)
    parser.add_argument("--mid-every", type=int, default=4)
    parser.add_argument("--late-every", type=int, default=20)
    args = parser.parse_args()

    if trainer_running():
        print("REFUSE: a DAST trainer is already running; one GPU job at a time.")
        return 2
    free, utilisation, others = accelerator_state()
    if free >= 0 and (free < args.min_free_mib or utilisation > args.max_busy_util):
        print(
            f"REFUSE: accelerator busy -- {free} MiB free, {utilisation}% utilised "
            f"(need >={args.min_free_mib} MiB and <={args.max_busy_util}%).\n"
            f"  other compute processes: {others or 'none reported'}"
        )
        return 2

    config = load_config(args.config)
    config.validate()
    if config.evaluation.do_generation:
        raise SystemExit(
            "B2-T configs must set do_generation: false. This run has no capability "
            "endpoint and its accuracy must not be produced, let alone read."
        )

    output_dir = config.resolved_path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = Path(args.out) if args.out else output_dir / "grad_trajectory.jsonl"
    if trajectory_path.exists():
        trajectory_path.unlink()

    print(
        f"arm={config.experiment.arm} seed={config.experiment.seed} "
        f"config_sha={config.canonical_sha256()[:12]}\n"
        f"accelerator: {free} MiB free, {utilisation}% utilised\n"
        f"streaming to {trajectory_path}",
        flush=True,
    )

    base, tokenizer, info = build_hf_components(config, for_training=True)
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
    if torch.cuda.is_available():
        model.to("cuda")

    handle = trajectory_path.open("a", encoding="utf-8")
    started = time.perf_counter()
    counter = {"n": 0}

    def sink(row: dict) -> None:
        # Per-module detail is large and only needed for the exploratory secondary
        # analysis, so it is kept but the file stays one row per measured step.
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        counter["n"] += 1
        if row.get("measured") and counter["n"] % 20 == 0:
            print(
                f"  step {row['global_step']:>5}  lora_cos {row['lora_only_cosine']:+.5f}"
                f"  lm {row['lm_loss']:.5f}  state {row['state_loss']:.5f}"
                f"  [{time.perf_counter() - started:.0f}s]",
                flush=True,
            )

    probe = TrajectoryProbe(
        sink,
        dense_until=args.dense_until,
        mid_until=args.mid_until,
        mid_every=args.mid_every,
        late_every=args.late_every,
    )
    try:
        result = train(model, tokenizer, config, model_info=info, step_probe=probe)
    finally:
        handle.close()

    measured = sum(1 for row in probe.rows if row.get("measured"))
    summary = {
        "arm": config.experiment.arm,
        "seed": config.experiment.seed,
        "config_sha256": config.canonical_sha256(),
        "config_path": str(args.config),
        "global_steps": result.global_steps,
        "optimizer_steps": result.optimizer_steps,
        "measured_steps": measured,
        "trajectory": str(trajectory_path),
        "elapsed_seconds": time.perf_counter() - started,
        "schedule": {
            "dense_until": args.dense_until, "mid_until": args.mid_until,
            "mid_every": args.mid_every, "late_every": args.late_every,
        },
        "claim_boundary": (
            "B2-T diagnosis only. No capability endpoint; generation disabled. DEV only, "
            "TEST sealed. Accuracy from this run must not be reported."
        ),
    }
    (output_dir / "grad_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"\ndone: {measured} measured steps of {result.global_steps} in "
        f"{summary['elapsed_seconds']:.0f}s\nwrote {trajectory_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
