"""Shared fixtures.

Tests that need the real dataset skip cleanly when it is not present, so the unit
suite runs in a bare checkout while the integration suite runs wherever the
dataset is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rineya_lab.config import DatasetConfig
from rineya_lab.data.loading import DatasetNotFoundError, RineyaDataset, resolve_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset_root() -> Path | None:
    hint = os.environ.get("RINEYA_DATASET_ROOT")
    candidates = [Path(hint)] if hint else []
    candidates += [
        REPO_ROOT.parent / "rineya_sequence_independence_dataset_v1",
        REPO_ROOT / "rineya_sequence_independence_dataset_v1",
    ]
    for candidate in candidates:
        if (candidate / "data" / "full_context" / "dev.jsonl").exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    root = _dataset_root()
    if root is None:
        pytest.skip("Rineya dataset not found; set RINEYA_DATASET_ROOT to enable dataset tests")
    return root


@pytest.fixture(scope="session")
def dataset(dataset_root: Path) -> RineyaDataset:
    try:
        paths = resolve_dataset(DatasetConfig(root=str(dataset_root), verify_hashes=False))
    except DatasetNotFoundError:  # pragma: no cover
        pytest.skip("Rineya dataset not resolvable")
    return RineyaDataset(paths, verify_hashes=False)


@pytest.fixture(scope="session")
def dev_trajectories(dataset: RineyaDataset):
    return dataset.trajectories("dev")


@pytest.fixture
def base_config(dataset_root: Path):
    from rineya_lab.config import load_config

    return load_config(
        None,
        extra={
            "dataset": {"root": str(dataset_root), "verify_hashes": False},
            "experiment": {"seed": 20260818},
        },
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: requires torch/transformers/peft")


def requires_torch():
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return pytest.mark.skip(reason="torch/transformers/peft not installed")
    return pytest.mark.torch
