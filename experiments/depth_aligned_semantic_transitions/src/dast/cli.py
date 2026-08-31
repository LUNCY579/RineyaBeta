"""Command-line entry point for DAST."""

from __future__ import annotations

import argparse
import json
from typing import Any

from dast.compare import compare_prediction_files
from dast.config import assert_split_allowed, load_config
from dast.data import build_examples, load_records, validate_dataset
from dast.hf import build_hf_components
from dast.model import DASTModel
from dast.operators import LowRankOperatorBank
from dast.smoke import run_smoke
from dast.training import set_global_seed, train


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def command_validate_data(args: argparse.Namespace) -> int:
    report = validate_dataset(args.path)
    _print_json(report.as_dict())
    return 0 if report.passed else 1


def command_inspect_alignment(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    split = args.split or config.data.train_split
    # The printed canonical/training targets ARE compact state text, and the last
    # one contains the final answer. Inspecting alignment on the sealed split
    # would therefore leak TEST answers to stdout, so it takes the same guard as
    # training and evaluation.
    assert_split_allowed(config, split)
    records = load_records(
        config.data_path_for_split(split),
        split=split,
        limit=args.limit,
    )
    examples = build_examples(
        records,
        arm=config.arm_spec,
        num_stages=config.num_stages,
        seed=config.experiment.seed,
    )
    _print_json(
        [
            {
                "problem_id": item.record.problem_id,
                "arm": config.experiment.arm,
                "checkpoint_fractions": list(config.model.checkpoint_fractions),
                "canonical_targets": list(item.trajectory.canonical_texts),
                "training_targets": list(item.trajectory.training_texts),
                "assignment": list(item.trajectory.assignment),
                "canonical_source_indices": list(item.trajectory.canonical_source_indices),
                "training_source_indices": list(item.trajectory.training_source_indices),
            }
            for item in examples
        ]
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    # Seed BEFORE anything draws from the global RNG. The LoRA adapter and the
    # shared projector are both initialised during construction below, so seeding
    # only inside train() left every arm's starting weights unreproducible and
    # made the matched-seed contract unenforceable.
    set_global_seed(
        config.experiment.seed,
        deterministic_algorithms=config.training.deterministic_algorithms,
    )
    base_model, tokenizer, model_info = build_hf_components(config, for_training=True)
    operator = None
    if config.operators.enabled:
        operator = LowRankOperatorBank(
            config.semantic.state_dim,
            config.operators.num_operators,
            config.operators.rank,
            router_temperature=config.operators.router_temperature,
            frozen_random=config.operators.frozen_random,
        )
    model = DASTModel(
        base_model,
        checkpoint_fractions=config.model.checkpoint_fractions,
        state_dim=config.semantic.state_dim,
        projector_hidden_dim=config.semantic.projector_hidden_dim,
        pooling=config.model.pooling,
        operator_bank=operator,
        feedback_enabled=config.model.feedback_enabled,
        feedback_scale=config.model.feedback_scale,
        checkpoint_readout=config.model.checkpoint_readout,
    )
    model_info["dast_checkpoint_layers"] = list(model.checkpoint_layers)
    model_info["dast_state_dim"] = config.semantic.state_dim
    model_info["dast_operator_enabled"] = config.operators.enabled
    # Both readout and target geometry change what the objective means, so a run
    # is not interpretable without them recorded alongside the resolved layers.
    model_info["dast_checkpoint_readout"] = model.checkpoint_readout
    model_info["dast_target_centering"] = config.semantic.target_centering
    model_info["dast_contrastive_active"] = bool(
        config.semantic.contrastive_weight > 0.0 and config.training.batch_size >= 2
    )
    result = train(model, tokenizer, config, model_info=model_info)
    _print_json(
        {
            "output_dir": str(result.output_dir),
            "global_steps": result.global_steps,
            "optimizer_steps": result.optimizer_steps,
            "final_metrics": result.final_metrics,
        }
    )
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    report = run_smoke(args.output_dir)
    _print_json(report)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    report = compare_prediction_files(
        args.baseline,
        args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        output_path=args.output,
    )
    _print_json(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dast",
        description="Depth-Aligned Semantic Transitions research harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate the reasoning-state JSONL contract")
    validate.add_argument("path")
    validate.set_defaults(function=command_validate_data)

    inspect = subparsers.add_parser("inspect-alignment", help="show exact targets assigned to each depth")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--split")
    inspect.add_argument("--limit", type=int, default=5)
    inspect.set_defaults(function=command_inspect_alignment)

    train_parser = subparsers.add_parser("train", help="run a real Hugging Face/LoRA DEV experiment")
    train_parser.add_argument("--config", required=True)
    train_parser.set_defaults(function=command_train)

    smoke = subparsers.add_parser("smoke", help="run the offline toy end-to-end qualification")
    smoke.add_argument("--output-dir")
    smoke.set_defaults(function=command_smoke)

    compare = subparsers.add_parser("compare", help="paired comparison of generation prediction files")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=5000)
    compare.add_argument("--seed", type=int, default=17)
    compare.add_argument("--output")
    compare.set_defaults(function=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
