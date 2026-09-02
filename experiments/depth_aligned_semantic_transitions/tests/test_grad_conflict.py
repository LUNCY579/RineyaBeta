"""B2 gradient-cosine helper: diagnosis only, no optimizer step."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import os

import pytest

import torch

from dast.grad_conflict import conflict_from_losses, vector_cosine


def test_opposed_losses_have_cosine_minus_one():
    weight = torch.nn.Linear(4, 4, bias=False)
    data = torch.randn(3, 4)
    predicted = weight(data)
    lm = predicted.pow(2).mean()
    state = -predicted.pow(2).mean()
    report = conflict_from_losses(lm, state, list(weight.named_parameters()))
    assert abs(report["global_cosine"] + 1.0) < 1e-5
    assert report["fraction_modules_negative_cosine"] == 1.0


def test_identical_losses_have_cosine_plus_one():
    weight = torch.nn.Linear(3, 3, bias=False)
    data = torch.randn(2, 3)
    loss = weight(data).pow(2).mean()
    report = conflict_from_losses(loss, loss, list(weight.named_parameters()))
    assert abs(report["global_cosine"] - 1.0) < 1e-5


def test_zero_grad_cosine_is_nan():
    assert math.isnan(vector_cosine(torch.zeros(3), torch.ones(3)))


def test_lora_only_cosine_reconstructs_the_concatenated_cosine() -> None:
    """The per-module recombination must equal the true cosine of the concatenation.

    B2's headline `global_cosine` includes the state_projector, where grad_lm is
    identically zero because L_LM does not depend on it. That inflates ||g_state||
    without touching the dot product and shrinks |cosine| toward 0, making every arm
    look equally conflict-free. The corrected figure is rebuilt from per-module
    norms and cosines, so that reconstruction has to be exact.
    """

    import importlib.util
    import math
    import sys
    from pathlib import Path

    import torch

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "gradient_conflict_compare", scripts / "gradient_conflict_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gradient_conflict_compare"] = module
    spec.loader.exec_module(module)

    torch.manual_seed(5)
    blocks = {"lora.a": (7, 1.0), "lora.b": (5, 1.0), "state_projector": (11, 0.0)}
    left_parts, right_parts, rows = [], [], []
    for name, (size, lm_scale) in blocks.items():
        left = torch.randn(size, dtype=torch.float64) * lm_scale
        right = torch.randn(size, dtype=torch.float64)
        left_norm, right_norm = float(left.norm()), float(right.norm())
        cosine = (
            float(torch.dot(left, right) / (left_norm * right_norm))
            if left_norm > 0 and right_norm > 0
            else float("nan")
        )
        rows.append(
            {"module": name, "cosine": cosine, "lm_norm": left_norm, "state_norm": right_norm}
        )
        if name != "state_projector":
            left_parts.append(left)
            right_parts.append(right)

    expected_left = torch.cat(left_parts)
    expected_right = torch.cat(right_parts)
    expected = float(
        torch.dot(expected_left, expected_right)
        / (expected_left.norm() * expected_right.norm())
    )
    assert module.lora_only_cosine({"modules": rows}) == pytest.approx(expected, abs=1e-12)

    # And it must genuinely differ from the diluted global figure.
    full_left = torch.cat([expected_left, torch.zeros(11, dtype=torch.float64)])
    full_right = torch.cat([expected_right, torch.randn(11, dtype=torch.float64)])
    diluted = float(
        torch.dot(full_left, full_right) / (full_left.norm() * full_right.norm())
    )
    assert abs(diluted) < abs(expected), "projector block must shrink |cosine|"
    assert math.copysign(1, diluted) == math.copysign(1, expected)


def test_step_probe_does_not_perturb_the_optimizer_trajectory(tmp_path) -> None:
    """The B2-T probe must be bit-identically inert.

    This is validity check 1 of PREREGISTRATION_B2_TRAJECTORY.md. If taking the
    diagnostic gradients moved the optimizer even slightly, the measured trajectory
    would describe a run nobody else could reproduce, and the arm comparison would be
    confounded by the act of measuring.

    torch.autograd.grad returns gradients without accumulating into .grad, and
    backward consumes no RNG, so identical parameters are the correct expectation --
    not merely close ones.
    """

    import copy

    from dast.smoke import _make_model, build_smoke_config
    from dast.training import train

    def run(with_probe: bool):
        config = build_smoke_config(tmp_path / ("probe" if with_probe else "plain"),
                                    arm="state_ordered")
        config.training.deterministic_algorithms = True
        config.evaluation.do_generation = False
        model, tokenizer = _make_model(config)
        # Identical initialisation for both runs: only the probe differs.
        model.load_state_dict(copy.deepcopy(_REFERENCE_STATE))
        probe = None
        rows: list[dict] = []
        if with_probe:
            from dast.grad_conflict import TrajectoryProbe

            probe = TrajectoryProbe(rows.append, dense_until=10**9)
        train(model, tokenizer, config, step_probe=probe)
        return (
            {name: value.detach().clone() for name, value in model.state_dict().items()},
            rows,
        )

    config = build_smoke_config(tmp_path / "seed", arm="state_ordered")
    seed_model, _ = _make_model(config)
    global _REFERENCE_STATE
    _REFERENCE_STATE = copy.deepcopy(seed_model.state_dict())

    plain, no_rows = run(False)
    probed, rows = run(True)

    assert no_rows == []
    assert rows, "the probe recorded nothing; the schedule or hook is wrong"
    assert any(row.get("measured") for row in rows), "no row carries a measurement"
    for name, value in plain.items():
        assert torch.equal(value, probed[name]), (
            f"parameter {name} diverged: the probe perturbed the optimizer trajectory"
        )


_REFERENCE_STATE: dict = {}


def test_trajectory_probe_schedule_is_dense_early_then_sparse() -> None:
    from dast.grad_conflict import TrajectoryProbe

    # Amended schedule, PREREGISTRATION_B2_TRAJECTORY.md section 9c.
    probe = TrajectoryProbe(None, dense_until=120, mid_until=400, mid_every=4, late_every=20)
    assert all(probe.should_measure(step) for step in range(1, 121))
    assert probe.should_measure(124) and not probe.should_measure(125)
    assert probe.should_measure(420) and not probe.should_measure(404)
    measured = sum(1 for step in range(1, 2001) if probe.should_measure(step))
    # Preregistered budget: about 270 of 2000 global steps.
    assert 250 <= measured <= 290, measured
    # The EARLY window must give 15 whole optimizer steps to block-bootstrap over.
    assert sum(1 for s in range(1, 121) if probe.should_measure(s)) == 120


def test_trajectory_probe_honours_an_explicit_schedule() -> None:
    from dast.grad_conflict import TrajectoryProbe

    probe = TrajectoryProbe(None, schedule=[3, 7, 11])
    assert [step for step in range(1, 20) if probe.should_measure(step)] == [3, 7, 11]


def test_diagnostic_grads_under_checkpointing_do_not_shift_dropout_masks() -> None:
    """The probe must not perturb RNG when gradient checkpointing recomputes.

    This is stronger than the toy-model non-interference test above, which exercises
    neither gradient checkpointing nor dropout -- the toy backbone has no LoRA and
    smoke does not enable checkpointing, so that test alone gives false assurance
    about the real run, where checkpointing is ON and LoRA dropout is 0.05.

    The risk is specific: if a checkpointed segment consumes RNG while being
    recomputed for the DIAGNOSTIC backward, the PRODUCTION backward that follows sees
    different dropout masks and the probe silently alters the optimizer trajectory.
    torch.utils.checkpoint defaults to preserve_rng_state=True, which saves and
    restores the RNG around recomputation, so the correct expectation is exact
    equality -- checked for both checkpoint implementations.
    """

    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint

    class Block(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = nn.Linear(width, width)
            self.dropout = nn.Dropout(0.05)

        def forward(self, value):
            return self.dropout(torch.relu(self.linear(value)))

    class Net(nn.Module):
        def __init__(self, width: int = 16, depth: int = 4, reentrant: bool = False) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
            self.head = nn.Linear(width, width)
            self.reentrant = reentrant

        def forward(self, value):
            for block in self.blocks:
                value = checkpoint(block, value, use_reentrant=self.reentrant)
            return self.head(value)

    def run(with_probe: bool, reentrant: bool, steps: int = 6):
        torch.manual_seed(0)
        net = Net(reentrant=reentrant)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        torch.manual_seed(1234)
        batches = [torch.randn(3, 16) for _ in range(steps)]
        torch.manual_seed(999)  # the dropout stream both variants must share
        for batch in batches:
            output = net(batch)
            lm_like = output.pow(2).mean()
            state_like = output.abs().mean()
            total = lm_like + state_like
            if with_probe:
                params = [p for p in net.parameters() if p.requires_grad]
                torch.autograd.grad(lm_like, params, retain_graph=True, allow_unused=True)
                torch.autograd.grad(state_like, params, retain_graph=True, allow_unused=True)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        return [p.detach().clone() for p in net.parameters()]

    for reentrant in (False, True):
        plain = run(False, reentrant)
        probed = run(True, reentrant)
        for index, (left, right) in enumerate(zip(plain, probed)):
            assert torch.equal(left, right), (
                f"use_reentrant={reentrant}: parameter {index} diverged, so the diagnostic "
                "backward shifted the dropout stream"
            )


def test_batched_module_statistics_match_the_readable_per_module_loop() -> None:
    """The device-sync optimisation must be numerically identical, not just close.

    conflict_from_losses computes ~197 module groups. Calling float() per statistic
    cost ~600 device syncs per measured step, 2.7s on top of a 0.63s training step.
    The batched version transfers once. Since it feeds a preregistered decision rule,
    equality is checked against a literal transcription of the loop it replaced.
    """

    import torch.nn as nn

    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(12, 12), nn.Tanh(), nn.Linear(12, 5))
    batch = torch.randn(6, 12)
    output = model(batch)
    lm_like = output.pow(2).mean()
    state_like = output.abs().mean()

    report = conflict_from_losses(lm_like, state_like, model.named_parameters())

    # Literal transcription of the replaced loop, from the same gradients.
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    grad_lm = torch.autograd.grad(lm_like, params, retain_graph=True, allow_unused=True)
    grad_state = torch.autograd.grad(state_like, params, retain_graph=True, allow_unused=True)
    from collections import defaultdict

    from dast.grad_conflict import _group_key, vector_cosine

    buckets: dict = defaultdict(lambda: {"lm": [], "state": []})
    for name, glm, gst, param in zip(names, grad_lm, grad_state, params):
        key = _group_key(name)
        buckets[key]["lm"].append(
            param.new_zeros(param.numel()) if glm is None else glm.detach().reshape(-1)
        )
        buckets[key]["state"].append(
            param.new_zeros(param.numel()) if gst is None else gst.detach().reshape(-1)
        )
    expected = {}
    for key, bucket in sorted(buckets.items()):
        left = torch.cat([item.float() for item in bucket["lm"]])
        right = torch.cat([item.float() for item in bucket["state"]])
        expected[key] = (
            vector_cosine(left, right), float(left.norm()), float(right.norm())
        )

    assert {row["module"] for row in report["modules"]} == set(expected)
    for row in report["modules"]:
        cosine, lm_norm, state_norm = expected[row["module"]]
        assert row["cosine"] == pytest.approx(cosine, rel=1e-6, abs=1e-9), row["module"]
        assert row["lm_norm"] == pytest.approx(lm_norm, rel=1e-6), row["module"]
        assert row["state_norm"] == pytest.approx(state_norm, rel=1e-6), row["module"]


def test_probe_raises_early_stop_after_the_requested_step() -> None:
    from dast.grad_conflict import EarlyStop, TrajectoryProbe

    rows: list[dict] = []

    class Breakdown:
        lm = torch.tensor(1.0)
        state = torch.tensor(1.0)  # requires_grad False -> recorded, grads skipped

    probe = TrajectoryProbe(rows.append, dense_until=10**9, max_global_step=3)
    for step in (1, 2):
        probe(global_step=step, optimizer_step=1, model=None, breakdown=Breakdown())
    with pytest.raises(EarlyStop) as raised:
        probe(global_step=3, optimizer_step=1, model=None, breakdown=Breakdown())
    assert raised.value.global_step == 3
    # The row for the stopping step must already be recorded, so the requested
    # window is complete rather than one step short.
    assert [row["global_step"] for row in rows] == [1, 2, 3]


def test_analyser_refuses_a_verdict_on_absent_or_truncated_data(tmp_path) -> None:
    """REFUTED must never be the default output of missing data.

    Found by adversarial review of the frozen protocol. The first analyser computed
    `refuted = all(not ci_excludes_zero)`, and a NaN interval yields
    ci_excludes_zero = False, so three EMPTY trajectories printed
    "H2-T REFUTED ... the DAST closure is complete" and wrote h2t_refuted: true.
    That is the verdict which closes the whole programme, and it was unfalsifiable by
    instrument failure: zero data, an all-NaN column and a genuine null were
    indistinguishable in the artifact.
    """

    import importlib.util
    import json as _json
    import subprocess

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    runs = tmp_path / "runs"
    for arm in ("state_ordered", "state_reverse", "state_final_repeated"):
        directory = runs / f"diag_b2traj_s29_{arm}"
        directory.mkdir(parents=True)
        (directory / "grad_trajectory.jsonl").write_text("", encoding="utf-8")

    out = tmp_path / "verdict.json"
    done = subprocess.run(
        [sys.executable, str(scripts / "gradient_trajectory_analyse.py"),
         "--runs", str(runs), "--out", str(out)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert done.returncode == 3, f"expected refusal exit 3, got {done.returncode}"
    assert "INSUFFICIENT COVERAGE" in done.stdout
    # No VERDICT line at all. The refusal text does mention the word REFUTED, in the
    # sentence explaining why it may not be emitted, so the check is on the verdict
    # channel rather than on the presence of the word.
    assert "VERDICT:" not in done.stdout

    payload = _json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "INSUFFICIENT_COVERAGE"
    assert payload["verdict"] is None
    assert "h2t_refuted" not in payload, (
        "no boolean a reader could mistake for a null result may be written"
    )


def test_block_bootstrap_raises_rather_than_returning_nan(tmp_path) -> None:
    """A degenerate interval must be an error, not a value that reads as measured."""

    import importlib.util

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "gradient_trajectory_analyse", scripts / "gradient_trajectory_analyse.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gradient_trajectory_analyse"] = module
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="at least 2 blocks"):
        module.block_bootstrap({1: 0.1, 2: 0.2})  # both in the same block of 8
    mean, low, high, blocks = module.block_bootstrap({1: 0.1, 9: 0.2, 17: 0.3})
    assert blocks == 3 and low <= mean <= high
