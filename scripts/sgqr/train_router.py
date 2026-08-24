#!/usr/bin/env python3
"""Train a state-conditioned SGQR significance router from utility traces.

This predicts group marginal utility directly. It does not estimate next-token
confidence and contains no confidence threshold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.sgqr.remnant import DynamicUtilityRouter
from scripts.sgqr.utility import utility_capture


def split_indices(count: int, dev_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 2:
        raise ValueError("router training needs at least two trace rows")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    dev_count = max(1, min(count - 1, int(round(count * dev_fraction))))
    return order[dev_count:], order[:dev_count]


def mean_rank_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Spearman-like rank correlation without scipy."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    pred_rank = torch.argsort(torch.argsort(prediction, dim=-1), dim=-1).float()
    target_rank = torch.argsort(torch.argsort(target, dim=-1), dim=-1).float()
    pred_rank = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
    target_rank = target_rank - target_rank.mean(dim=-1, keepdim=True)
    numerator = (pred_rank * target_rank).sum(dim=-1)
    denominator = pred_rank.norm(dim=-1) * target_rank.norm(dim=-1)
    value = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
    return float(value.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--budget-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload = torch.load(args.trace, map_location="cpu", weights_only=True)
    features = payload["features"].float()
    utilities = payload["utilities"].float()
    metadata = dict(payload.get("metadata", {}))
    if features.ndim != 2 or utilities.ndim != 2:
        raise ValueError("trace features and utilities must be rank-2")
    if features.shape[0] != utilities.shape[0]:
        raise ValueError("feature and utility row counts differ")

    count, hidden_size = features.shape
    group_count = utilities.shape[1]
    budget_groups = args.budget_groups or max(1, group_count // 2)
    if budget_groups < 1 or budget_groups > group_count:
        raise ValueError("budget-groups outside valid range")

    train_idx, dev_idx = split_indices(count, args.dev_fraction, args.seed)
    feature_mean = features[train_idx].mean(dim=0)
    feature_scale = features[train_idx].std(dim=0).clamp_min(1e-5)
    utility_mean = utilities[train_idx].mean(dim=0)
    utility_scale = utilities[train_idx].std(dim=0).clamp_min(1e-6)

    normalized_features = (features - feature_mean) / feature_scale
    normalized_utilities = (utilities - utility_mean) / utility_scale

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DynamicUtilityRouter(hidden_size, group_count, args.rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)

    best_state = None
    best_dev = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation = train_idx[torch.randperm(len(train_idx), generator=generator)]
        train_losses = []
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            x = normalized_features[indices].to(device)
            target = normalized_utilities[indices].to(device)
            prediction = model(x)
            weights = torch.where(
                utilities[indices].to(device) > 0,
                torch.full_like(target, args.positive_weight),
                torch.ones_like(target),
            )
            loss = ((prediction - target).pow(2) * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.inference_mode():
            dev_prediction = model(normalized_features[dev_idx].to(device))
            dev_target = normalized_utilities[dev_idx].to(device)
            dev_loss = float(F.mse_loss(dev_prediction, dev_target).item())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "dev_loss": dev_loss,
            }
        )
        if dev_loss < best_dev:
            best_dev = dev_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("router training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        dev_prediction_norm = model(normalized_features[dev_idx].to(device)).cpu()
    dev_prediction = dev_prediction_norm * utility_scale + utility_mean
    dev_target = utilities[dev_idx]
    selected = torch.topk(dev_prediction, k=budget_groups, dim=-1).indices
    capture = utility_capture(dev_target, selected)

    random_generator = torch.Generator().manual_seed(args.seed + 1)
    random_scores = torch.rand(dev_prediction.shape, generator=random_generator)
    random_selected = torch.topk(random_scores, k=budget_groups, dim=-1).indices
    random_capture = utility_capture(dev_target, random_selected)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "sgqr-utility-router-v1",
        "state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "utility_mean": utility_mean,
        "utility_scale": utility_scale,
        "metadata": {
            **metadata,
            "rank": args.rank,
            "hidden_size": hidden_size,
            "group_count": group_count,
            "budget_groups": budget_groups,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, out_path)

    report = {
        "format": "sgqr-router-training-report-v1",
        "trace": str(args.trace),
        "rows": count,
        "train_rows": int(len(train_idx)),
        "dev_rows": int(len(dev_idx)),
        "rank": args.rank,
        "budget_groups": budget_groups,
        "best_dev_normalized_mse": best_dev,
        "dev_rank_correlation": mean_rank_correlation(dev_prediction, dev_target),
        "dev_utility_capture": float(capture.mean().item()),
        "random_utility_capture": float(random_capture.mean().item()),
        "utility_capture_advantage": float(
            capture.mean().item() - random_capture.mean().item()
        ),
        "checkpoint": str(out_path),
        "history": history,
    }
    report_path = out_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
