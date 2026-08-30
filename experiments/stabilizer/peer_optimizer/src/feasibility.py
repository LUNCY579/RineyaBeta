#!/usr/bin/env python3
"""Controlled feasibility test for peer-conditioned sparse native updates.

This is deliberately a synthetic quadratic study, not evidence that the method
improves Rineya.  It asks only whether the implemented mathematics can learn to
use peer matrices as a denoising prior when a tiny calibration batch gives a
noisy task gradient.  The hidden optimum is constructed from peer structural
properties, so the study has a known ground truth and meaningful negative
controls.

Usage:
    python feasibility.py
    python feasibility.py --train-steps 160 --eval-tasks 96 --out ../results/feasibility.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Callable

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

from peer_stabilizer import (  # noqa: E402
    LayerMetadata,
    LayerView,
    PeerStabilizerMatrix,
    exact_topk_mask,
    matrix_property_map,
    one_step_weight,
)


@dataclass
class SyntheticTask:
    layers: list[LayerView]
    target_index: int
    initial_weight: Tensor
    optimum_weight: Tensor
    true_delta: Tensor
    observed_gradient: Tensor
    budget: int


@dataclass
class Aggregate:
    mean_loss: float
    std_loss: float
    mean_support_recall: float
    std_support_recall: float
    mean_improvement: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _peer_property_consensus(peers: list[Tensor]) -> Tensor:
    maps = torch.stack([matrix_property_map(p).mean(dim=0) for p in peers], dim=0)
    return maps.mean(dim=0)


def make_task(
    generator: torch.Generator,
    *,
    layers: int,
    rows: int,
    cols: int,
    budget: int,
    gradient_noise: float,
    peer_noise: float,
) -> SyntheticTask:
    if layers < 3:
        raise ValueError("synthetic study needs at least two peers and one target")

    latent = torch.randn(rows, cols, generator=generator)
    # Structured row/column energy makes the peer property maps informative while
    # preserving task-to-task variation.
    row_scale = torch.exp(0.55 * torch.randn(rows, generator=generator))
    col_scale = torch.exp(0.55 * torch.randn(cols, generator=generator))
    latent = latent * row_scale[:, None] * col_scale[None, :]

    peer_weights: list[Tensor] = []
    for _ in range(layers - 1):
        scale = 0.75 + 0.50 * torch.rand((), generator=generator)
        peer_weights.append(scale * latent + peer_noise * torch.randn(rows, cols, generator=generator))

    consensus = _peer_property_consensus(peer_weights)
    true_mask = exact_topk_mask(consensus, budget)
    signed_source = torch.sign(latent)
    signed_source = torch.where(signed_source == 0, torch.ones_like(signed_source), signed_source)
    magnitudes = 0.55 + 0.45 * consensus / consensus.max().clamp_min(1e-8)
    true_delta = true_mask * signed_source * magnitudes

    initial = 0.10 * torch.randn(rows, cols, generator=generator)
    optimum = initial + true_delta
    true_gradient = initial - optimum
    observed_gradient = true_gradient + gradient_noise * torch.randn(
        rows, cols, generator=generator
    )

    views: list[LayerView] = []
    for depth, peer in enumerate(peer_weights):
        views.append(
            LayerView(
                weight=peer,
                metadata=LayerMetadata(
                    name=f"peer_{depth}", depth=depth, module_type="mlp.down_proj"
                ),
            )
        )
    target_index = len(views)
    views.append(
        LayerView(
            weight=initial,
            gradient=observed_gradient,
            metadata=LayerMetadata(
                name="target", depth=target_index, module_type="mlp.down_proj"
            ),
        )
    )
    return SyntheticTask(
        layers=views,
        target_index=target_index,
        initial_weight=initial,
        optimum_weight=optimum,
        true_delta=true_delta,
        observed_gradient=observed_gradient,
        budget=budget,
    )


def true_loss(weight: Tensor, optimum: Tensor) -> Tensor:
    return 0.5 * (weight - optimum).square().mean()


def support_recall(mask: Tensor, true_delta: Tensor) -> float:
    truth = true_delta != 0
    return float((mask.bool() & truth).sum().item() / truth.sum().item())


def _common_shrinkage(noise: float) -> float:
    # For signal variance near one and additive gradient noise sigma^2, this is the
    # scalar minimum-MSE shrinkage.  All non-oracle controls use the same value.
    return 1.0 / (1.0 + noise**2)


def evaluate_arm(
    task: SyntheticTask,
    arm: str,
    model: PeerStabilizerMatrix,
    rng: torch.Generator,
    shrinkage: float,
) -> tuple[float, float, float]:
    initial_loss = float(true_loss(task.initial_weight, task.optimum_weight).item())
    gradient = task.observed_gradient

    if arm == "peer_conditioned":
        with torch.no_grad():
            output = model(
                task.layers,
                task.target_index,
                task.budget,
                task_gradient=gradient,
                straight_through=False,
            )
            # The generated scale is bounded.  Multiplying by the common
            # shrinkage keeps noise treatment matched to the other non-oracle arms.
            updated = task.initial_weight - shrinkage * output.hard_mask * output.step_scale * gradient
            mask = output.hard_mask
    elif arm == "gradient_topk":
        mask = exact_topk_mask(gradient.abs(), task.budget)
        updated = task.initial_weight - shrinkage * mask * gradient
    elif arm == "target_only":
        target_score = gradient.abs() * matrix_property_map(task.initial_weight)[0]
        mask = exact_topk_mask(target_score, task.budget)
        updated = task.initial_weight - shrinkage * mask * gradient
    elif arm == "random":
        flat = torch.rand(task.initial_weight.numel(), generator=rng)
        mask = exact_topk_mask(flat.reshape_as(task.initial_weight), task.budget)
        updated = task.initial_weight - shrinkage * mask * gradient
    elif arm == "shuffled_peers":
        shuffled = list(task.layers)
        peer_indices = list(range(task.target_index))
        perm = torch.randperm(len(peer_indices), generator=rng).tolist()
        # Destroy coordinate-level coworker information while retaining each peer's
        # marginal value distribution and metadata.
        damaged: list[LayerView] = []
        for destination, source in zip(peer_indices, perm):
            w = shuffled[source].weight.reshape(-1)
            coord_perm = torch.randperm(w.numel(), generator=rng)
            damaged_weight = w[coord_perm].reshape_as(shuffled[destination].weight)
            damaged.append(
                LayerView(damaged_weight, shuffled[destination].metadata)
            )
        damaged.append(shuffled[task.target_index])
        with torch.no_grad():
            output = model(
                damaged,
                task.target_index,
                task.budget,
                task_gradient=gradient,
                straight_through=False,
            )
            mask = output.hard_mask
            updated = task.initial_weight - shrinkage * mask * output.step_scale * gradient
    elif arm == "dense":
        mask = torch.ones_like(task.initial_weight)
        updated = task.initial_weight - shrinkage * gradient
    elif arm == "oracle":
        mask = task.true_delta.ne(0).to(task.initial_weight.dtype)
        updated = task.optimum_weight
    elif arm == "no_update":
        mask = torch.zeros_like(task.initial_weight)
        updated = task.initial_weight
    else:
        raise ValueError(f"unknown arm: {arm}")

    loss = float(true_loss(updated, task.optimum_weight).item())
    recall = support_recall(mask, task.true_delta) if arm != "no_update" else 0.0
    return loss, recall, initial_loss - loss


def train_generator(
    model: PeerStabilizerMatrix,
    task_factory: Callable[[], SyntheticTask],
    *,
    train_steps: int,
    meta_batch: int,
    learning_rate: float,
    shrinkage: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    curve: list[float] = []
    model.train()
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        for _ in range(meta_batch):
            task = task_factory()
            output = model(
                task.layers,
                task.target_index,
                task.budget,
                task_gradient=task.observed_gradient,
                straight_through=True,
                mask_temperature=0.30,
            )
            updated = one_step_weight(
                task.initial_weight,
                task.observed_gradient,
                output,
                learning_rate=shrinkage,
                training=True,
            )
            losses.append(true_loss(updated, task.optimum_weight))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        curve.append(float(loss.detach().item()))
    return curve


def aggregate(rows: list[tuple[float, float, float]]) -> Aggregate:
    losses = torch.tensor([r[0] for r in rows])
    recalls = torch.tensor([r[1] for r in rows])
    improvements = torch.tensor([r[2] for r in rows])
    return Aggregate(
        mean_loss=float(losses.mean().item()),
        std_loss=float(losses.std(unbiased=False).item()),
        mean_support_recall=float(recalls.mean().item()),
        std_support_recall=float(recalls.std(unbiased=False).item()),
        mean_improvement=float(improvements.mean().item()),
    )


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    eval_rng = torch.Generator().manual_seed(args.seed + 2)
    control_rng = torch.Generator().manual_seed(args.seed + 3)

    def make_train() -> SyntheticTask:
        return make_task(
            train_rng,
            layers=args.layers,
            rows=args.rows,
            cols=args.cols,
            budget=args.budget,
            gradient_noise=args.gradient_noise,
            peer_noise=args.peer_noise,
        )

    model = PeerStabilizerMatrix(hidden_dim=args.hidden_dim)
    shrinkage = _common_shrinkage(args.gradient_noise)
    curve = train_generator(
        model,
        make_train,
        train_steps=args.train_steps,
        meta_batch=args.meta_batch,
        learning_rate=args.meta_lr,
        shrinkage=shrinkage,
    )

    arms = (
        "no_update",
        "random",
        "target_only",
        "gradient_topk",
        "shuffled_peers",
        "peer_conditioned",
        "dense",
        "oracle",
    )
    results: dict[str, list[tuple[float, float, float]]] = {a: [] for a in arms}
    model.eval()
    attention_rows: list[list[float]] = []
    feature_rows: list[list[float]] = []
    for _ in range(args.eval_tasks):
        task = make_task(
            eval_rng,
            layers=args.layers,
            rows=args.rows,
            cols=args.cols,
            budget=args.budget,
            gradient_noise=args.gradient_noise,
            peer_noise=args.peer_noise,
        )
        for arm in arms:
            results[arm].append(evaluate_arm(task, arm, model, control_rng, shrinkage))
        with torch.no_grad():
            out = model(
                task.layers,
                task.target_index,
                task.budget,
                task_gradient=task.observed_gradient,
                straight_through=False,
            )
            attention_rows.append(out.attention.tolist())
            feature_rows.append(out.feature_weights.tolist())

    metrics = {arm: asdict(aggregate(rows)) for arm, rows in results.items()}
    peer = metrics["peer_conditioned"]
    grad = metrics["gradient_topk"]
    shuffled = metrics["shuffled_peers"]
    random_arm = metrics["random"]

    checks = {
        "peer_beats_random_loss": peer["mean_loss"] < random_arm["mean_loss"],
        "peer_beats_gradient_topk_loss": peer["mean_loss"] < grad["mean_loss"],
        "peer_beats_shuffled_peers_loss": peer["mean_loss"] < shuffled["mean_loss"],
        "peer_support_recall_beats_random": (
            peer["mean_support_recall"] > random_arm["mean_support_recall"]
        ),
        "meta_loss_decreased": curve[-1] < curve[0],
    }

    attention = torch.tensor(attention_rows)
    features = torch.tensor(feature_rows)
    report = {
        "status": "synthetic_mathematical_feasibility_only",
        "claim_boundary": (
            "This controlled study validates implementation mechanics and the "
            "possibility of learning a peer prior. It is not an LLM quality result."
        ),
        "seed": args.seed,
        "configuration": {
            "layers": args.layers,
            "shape": [args.rows, args.cols],
            "budget": args.budget,
            "budget_fraction": args.budget / (args.rows * args.cols),
            "gradient_noise": args.gradient_noise,
            "peer_noise": args.peer_noise,
            "train_steps": args.train_steps,
            "meta_batch": args.meta_batch,
            "meta_lr": args.meta_lr,
            "eval_tasks": args.eval_tasks,
            "common_gradient_shrinkage": shrinkage,
        },
        "metrics": metrics,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "meta_curve": {
            "first": curve[0],
            "last": curve[-1],
            "minimum": min(curve),
            "points": curve,
        },
        "mean_attention": attention.mean(dim=0).tolist(),
        "mean_feature_weights": features.mean(dim=0).tolist(),
        "feature_order": [
            "task_gradient_magnitude",
            "peer_element_magnitude",
            "peer_row_energy",
            "peer_column_energy",
            "peer_joint_leverage",
            "target_element_magnitude",
            "peer_target_disagreement",
            "gradient_peer_interaction",
        ],
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--gradient-noise", type=float, default=1.35)
    ap.add_argument("--peer-noise", type=float, default=0.20)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--train-steps", type=int, default=140)
    ap.add_argument("--meta-batch", type=int, default=6)
    ap.add_argument("--meta-lr", type=float, default=3e-3)
    ap.add_argument("--eval-tasks", type=int, default=96)
    ap.add_argument("--out", type=Path, default=EXP / "results" / "feasibility.json")
    args = ap.parse_args()

    if args.budget > args.rows * args.cols:
        ap.error("budget exceeds matrix size")
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Peer-conditioned stabilizer matrix: synthetic feasibility")
    print(f"status: {report['status']}")
    print(f"all checks pass: {report['all_checks_pass']}")
    print("\narm                 loss       recall    improvement")
    for arm, row in report["metrics"].items():
        print(
            f"{arm:20s} {row['mean_loss']:.6f}  "
            f"{row['mean_support_recall']:.4f}    {row['mean_improvement']:+.6f}"
        )
    print(f"\nwrote {args.out}")
    if not report["all_checks_pass"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        print("failed checks: " + ", ".join(failed))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
