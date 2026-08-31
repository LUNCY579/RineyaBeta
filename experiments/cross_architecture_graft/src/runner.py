"""Command-line entry point for planning, smoke proof, and local real runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evidence import atomic_write_json, build_manifest
from .factory import component_plan
from .integration import save_graft_pack
from .real_runner import run_real
from .smoke import run_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="strict JSON experiment config")
    parser.add_argument(
        "--command",
        choices=("preflight", "smoke", "real"),
        required=True,
    )
    parser.add_argument(
        "--out",
        help="manifest path for preflight/smoke; real mode uses output.run_dir",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan = component_plan(config)

    if args.command == "preflight":
        payload = {
            "status": "PREFLIGHT_PASS",
            "experiment_id": config.experiment_id,
            "config_sha256": config.sha256,
            "mode": config.mode,
            "sealed_evaluation_opened": False,
            "component_plan": plan,
        }
        if args.out:
            atomic_write_json(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "smoke":
        outcome = run_smoke(config)
        manifest = build_manifest(
            config=config,
            status=outcome.status,
            phase="synthetic_end_to_end_smoke",
            results={"component_plan": plan, **outcome.results},
            arms=outcome.arms,
            invariants=outcome.invariants,
        )
        output = Path(args.out or "experiments/cross_architecture_graft/results/smoke.json")
        atomic_write_json(output, manifest)
        save_graft_pack(
            outcome.candidate,
            output.parent / "smoke_graft_pack",
            {
                "experiment_id": config.experiment_id,
                "config_sha256": config.sha256,
                "status": outcome.status,
                "claim_boundary": manifest["claim_boundary"],
            },
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if outcome.status == "INFRASTRUCTURE_PASS_ONLY" else 1

    manifest, path = run_real(config)
    print(json.dumps({"manifest": str(path), "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
