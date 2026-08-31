"""Deterministic CPU end-to-end proof for the graft pipeline.

The synthetic teacher uses the same frozen foreign block later wired into the
candidate.  This proves plumbing, trainability boundaries, causal sequence
mixing, matched-control accounting, exportability, and exact zero-gate bypass.
It is deliberately not treated as evidence that the mechanism helps Rineya.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .bridges import LinearBridge
from .config import ExperimentConfig
from .controls import arm_record, build_parameter_matched_mlp_control
from .donors import build_donor
from .graft import (
    GraftBranch,
    GraftedDecoderLayer,
    ResidualGate,
    configure_graft_training,
    parameter_counts,
)


class ToyHostLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d_model))

    def forward(self, hidden_states: Tensor, **_: Any) -> Tensor:
        return hidden_states + 0.1 * torch.tanh(self.proj(hidden_states))


@dataclass
class SmokeOutcome:
    status: str
    results: dict[str, Any]
    arms: list[dict[str, Any]]
    invariants: dict[str, Any]
    candidate: GraftedDecoderLayer


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mse(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(F.mse_loss(model(inputs), targets).cpu())


def _fit(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> list[float]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = F.mse_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def run_smoke(config: ExperimentConfig) -> SmokeOutcome:
    raw = config.raw
    if config.mode != "smoke":
        raise ValueError("run_smoke requires mode='smoke'")

    seed = config.seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    host_dim = int(raw["host"]["hidden_size"])
    foreign_dim = int(raw["graft"]["foreign_dim"])
    if host_dim != foreign_dim:
        raise ValueError("the v1 synthetic smoke requires host_dim == foreign_dim")
    training = raw.get("training", {})
    steps = int(training.get("steps", 80))
    lr = float(training.get("lr", 3e-2))
    weight_decay = float(training.get("weight_decay", 0.0))
    grad_clip = float(training.get("grad_clip", 1.0))
    batch_size = int(training.get("batch_size", 64))
    seq_len = int(training.get("seq_len", 8))

    host = ToyHostLayer(host_dim)
    teacher = build_donor(config.donor_spec, foreign_dim)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    train_inputs = torch.randn(batch_size, seq_len, host_dim, generator=generator)
    dev_inputs = torch.randn(max(16, batch_size // 2), seq_len, host_dim, generator=generator)
    teacher_scale = 0.35
    with torch.no_grad():
        train_targets = host(train_inputs) + teacher_scale * teacher(train_inputs)
        dev_targets = host(dev_inputs) + teacher_scale * teacher(dev_inputs)

    candidate_donor = copy.deepcopy(teacher)
    candidate_branch = GraftBranch(
        host_dim=host_dim,
        foreign_dim=foreign_dim,
        input_bridge=LinearBridge(host_dim, foreign_dim, identity_init=True),
        donor=candidate_donor,
        output_bridge=LinearBridge(foreign_dim, host_dim, identity_init=True),
        gate=ResidualGate(host_dim, config.gate_spec),
    )
    candidate = GraftedDecoderLayer(
        copy.deepcopy(host),
        candidate_branch,
        exact_eval_bypass=bool(raw["graft"].get("exact_eval_bypass", True)),
    )
    configure_graft_training(candidate, freeze_host=True, freeze_donor=True)

    candidate.eval()
    with torch.no_grad():
        host_reference = candidate.host_layer(dev_inputs)
        initial_output = candidate(dev_inputs)
    max_abs_initial = float((initial_output - host_reference).abs().max().cpu())
    bit_exact_initial = torch.equal(initial_output, host_reference)

    candidate_counts = parameter_counts(candidate)
    torch.manual_seed(seed + 2)
    control, match = build_parameter_matched_mlp_control(
        host_layer=copy.deepcopy(host),
        d_model=host_dim,
        target_trainable_parameters=candidate_counts["trainable"],
        gate_spec=config.gate_spec,
    )

    host_before = _module_digest(candidate.host_layer)
    donor_before = _module_digest(candidate.graft_branch.donor)
    baseline_dev = _mse(candidate.host_layer, dev_inputs, dev_targets)
    candidate_initial_dev = _mse(candidate, dev_inputs, dev_targets)
    control_initial_dev = _mse(control, dev_inputs, dev_targets)

    candidate_curve = _fit(
        candidate,
        train_inputs,
        train_targets,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    control_curve = _fit(
        control,
        train_inputs,
        train_targets,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    candidate_final_dev = _mse(candidate, dev_inputs, dev_targets)
    control_final_dev = _mse(control, dev_inputs, dev_targets)

    host_unchanged = host_before == _module_digest(candidate.host_layer)
    donor_unchanged = donor_before == _module_digest(candidate.graft_branch.donor)
    candidate_improved = candidate_final_dev < candidate_initial_dev * 0.25
    finite = all(
        torch.isfinite(torch.tensor(value))
        for value in (
            baseline_dev,
            candidate_initial_dev,
            control_initial_dev,
            candidate_final_dev,
            control_final_dev,
        )
    )
    match_within_ten_percent = match.relative_gap <= 0.10
    passed = all(
        (
            bit_exact_initial,
            max_abs_initial == 0.0,
            host_unchanged,
            donor_unchanged,
            candidate_improved,
            finite,
            match_within_ten_percent,
        )
    )

    results = {
        "synthetic_teacher_scale": teacher_scale,
        "baseline_dev_mse": baseline_dev,
        "candidate_initial_dev_mse": candidate_initial_dev,
        "candidate_final_dev_mse": candidate_final_dev,
        "control_initial_dev_mse": control_initial_dev,
        "control_final_dev_mse": control_final_dev,
        "candidate_train_loss_first": candidate_curve[0],
        "candidate_train_loss_last": candidate_curve[-1],
        "control_train_loss_first": control_curve[0],
        "control_train_loss_last": control_curve[-1],
        "candidate_gate": candidate.graft_branch.last_stats,
        "matched_control": {
            "intermediate_size": match.intermediate_size,
            "relative_parameter_gap": match.relative_gap,
        },
        "interpretation": (
            "Synthetic wiring proof only: the candidate is given the exact frozen "
            "teacher block. Its result cannot support a model-quality or donor-knowledge claim."
        ),
    }
    invariants = {
        "zero_gate_bit_exact": bit_exact_initial,
        "zero_gate_max_abs_error": max_abs_initial,
        "host_parameters_unchanged": host_unchanged,
        "frozen_donor_parameters_unchanged": donor_unchanged,
        "candidate_training_reduced_dev_mse_by_4x": candidate_improved,
        "all_metrics_finite": bool(finite),
        "matched_control_parameter_gap_le_10pct": match_within_ten_percent,
        "sealed_evaluation_opened": False,
    }
    arms = [
        arm_record("host_baseline", candidate.host_layer, role="frozen host"),
        arm_record(
            "foreign_block_graft",
            candidate,
            role="candidate",
            donor=config.donor_spec.kind,
            frozen_donor=True,
        ),
        arm_record(
            "parameter_matched_swiglu",
            control,
            role="ordinary-capacity control",
            intermediate_size=match.intermediate_size,
        ),
    ]
    return SmokeOutcome(
        status="INFRASTRUCTURE_PASS_ONLY" if passed else "INFRASTRUCTURE_FAIL",
        results=results,
        arms=arms,
        invariants=invariants,
        candidate=candidate,
    )
