"""The factorised permutation scorer must equal a naive direct fit, exactly.

`_score_permutations` shares one Cholesky factorisation across all permutations
and reconstructs each right-hand side from precomputed blocks. That is a real
algebraic shortcut, and a silent error in it would produce a wrong scientific
ranking that looks perfectly plausible. So the contract is checked against a
literal transcription of the original section 2b estimator.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F

from dast.depth_probe import (
    center_targets,
    kendall_tau,
    layer_stage_profile,
    permutation_landscape,
    structured_permutations,
    _score_permutations,
    _solve_blocks,
    split_indices,
)


def _naive_fit_and_score(
    hidden_train: torch.Tensor,
    targets_train: torch.Tensor,
    hidden_test: torch.Tensor,
    targets_test: torch.Tensor,
    perm: tuple[int, ...],
    ridge: float,
) -> dict[str, float]:
    """Literal transcription of scripts/permutation_preference.py::fit_and_score."""

    stages = hidden_train.shape[1]
    width = hidden_train.shape[2]
    eye = torch.eye(stages, dtype=torch.bool)

    permuted_train = targets_train[:, list(perm), :]
    permuted_test = targets_test[:, list(perm), :]

    design = torch.cat(
        [hidden_train.reshape(-1, width), torch.ones(hidden_train.shape[0] * stages, 1, dtype=hidden_train.dtype)],
        dim=1,
    )
    weights = torch.linalg.solve(
        design.T @ design + ridge * torch.eye(design.shape[1], dtype=design.dtype),
        design.T @ permuted_train.reshape(-1, permuted_train.shape[-1]),
    )
    flat_test = torch.cat(
        [hidden_test.reshape(-1, width), torch.ones(hidden_test.shape[0] * stages, 1, dtype=hidden_test.dtype)],
        dim=1,
    )
    predicted = F.normalize(
        (flat_test @ weights).reshape(hidden_test.shape[0], stages, -1), dim=-1
    )
    actual = F.normalize(permuted_test, dim=-1)
    similarity = torch.einsum("nkd,njd->nkj", predicted, actual)
    diagonal = similarity[:, eye].mean().item()
    return {
        "L_state": 1.0 - diagonal,
        "diagonal_cosine": diagonal,
        "order_margin": diagonal - similarity[:, ~eye].mean().item(),
        "top1": (similarity.argmax(2) == torch.arange(stages)).float().mean().item(),
    }


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_factorised_scorer_matches_naive_fit(stages: int) -> None:
    torch.manual_seed(7)
    count, width, state_dim, ridge = 40, 11, 6, 1.0
    hidden = torch.randn(count, stages, width, dtype=torch.float64)
    targets = center_targets(torch.randn(count, stages, state_dim, dtype=torch.float64))

    train_idx, test_idx = split_indices(count, None)
    perms = list(itertools.permutations(range(stages)))
    factor, blocks = _solve_blocks(hidden[train_idx], targets[train_idx], ridge)
    fast = _score_permutations(
        factor, blocks, hidden[test_idx], targets[test_idx], perms, batch=3
    )

    for perm, scored in zip(perms, fast):
        expected = _naive_fit_and_score(
            hidden[train_idx], targets[train_idx],
            hidden[test_idx], targets[test_idx], perm, ridge,
        )
        got = scored.as_dict()
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, abs=1e-9), f"{perm} {key}"


def test_centering_matches_training_formulation() -> None:
    """center_targets must equal dast.training._encode_targets' within_problem path."""

    torch.manual_seed(3)
    grid = F.normalize(torch.randn(9, 4, 8), dim=-1)
    expected = F.normalize(grid - grid.mean(dim=1, keepdim=True), dim=-1)
    assert torch.allclose(center_targets(grid), expected, atol=1e-12)


def test_centered_targets_are_rank_deficient() -> None:
    """K centred vectors span at most K-1 dimensions. Stated so it is not a surprise."""

    torch.manual_seed(11)
    grid = F.normalize(torch.randn(1, 4, 16, dtype=torch.float64), dim=-1)
    centred = grid - grid.mean(dim=1, keepdim=True)
    assert torch.linalg.matrix_rank(centred[0], tol=1e-10).item() == 3
    # Renormalising rescales rows, so the SPAN is unchanged even though the rows
    # no longer sum to zero.
    renormalised = F.normalize(centred, dim=-1)
    assert torch.linalg.matrix_rank(renormalised[0], tol=1e-10).item() == 3


def test_permutation_landscape_is_exhaustive_and_ranks_identity_first_when_planted() -> None:
    """With a planted forward structure the identity must win. Guards metric sign."""

    torch.manual_seed(5)
    count, stages, width, state_dim = 80, 4, 12, 5
    targets = center_targets(torch.randn(count, stages, state_dim, dtype=torch.float64))
    # Build hidden states so that ONE shared linear map recovers stage k at depth k.
    projector = torch.randn(state_dim, width, dtype=torch.float64)
    hidden = targets @ projector + 0.01 * torch.randn(count, stages, width, dtype=torch.float64)

    result = permutation_landscape(hidden, targets, splits=4, seed=1)
    assert result["exhaustive"] is True
    assert result["num_permutations"] == 24
    assert result["summary"]["best_perm"] == [0, 1, 2, 3]
    assert result["summary"]["identity_rank_median"] == 1


def test_permutation_landscape_flat_when_targets_carry_no_depth_structure() -> None:
    """If hidden states are independent of stage, no assignment should dominate."""

    torch.manual_seed(13)
    count, stages, width, state_dim = 120, 4, 10, 5
    targets = center_targets(torch.randn(count, stages, state_dim, dtype=torch.float64))
    # Hidden state depends on the PROBLEM only, identical at every depth.
    shared = torch.randn(count, 1, width, dtype=torch.float64).expand(count, stages, width)
    result = permutation_landscape(hidden=shared.contiguous(), targets=targets, splits=4, seed=2)
    spread = result["summary"]["L_state_spread_mean"]
    assert spread < 1e-6, f"expected a flat landscape, got spread {spread}"


def test_structured_permutations_are_valid_permutations() -> None:
    for stages in (2, 4, 5, 6, 8):
        for name, perm in structured_permutations(stages).items():
            assert sorted(perm) == list(range(stages)), f"{name} at K={stages} is not a permutation"


def test_kendall_tau_endpoints() -> None:
    assert kendall_tau((0, 1, 2, 3)) == pytest.approx(1.0)
    assert kendall_tau((3, 2, 1, 0)) == pytest.approx(-1.0)


def test_layer_stage_profile_recovers_a_planted_diagonal() -> None:
    """Plant stage k decodable only at layer k; the profile must find the diagonal."""

    torch.manual_seed(17)
    count, stages, width, state_dim = 90, 4, 9, 4
    targets = center_targets(torch.randn(count, stages, state_dim, dtype=torch.float64))
    projector = torch.randn(state_dim, width, dtype=torch.float64)
    # layer j carries stage j's target, plus noise.
    hidden = torch.stack(
        [targets[:, stage] @ projector for stage in range(stages)], dim=1
    ) + 0.01 * torch.randn(count, stages, width, dtype=torch.float64)

    result = layer_stage_profile(hidden, targets, layers=[3, 7, 11, 15], splits=3, seed=4)
    assert result["best_layer_per_stage"] == [3, 7, 11, 15]
    assert result["monotone_in_stage"] is True
    assert result["spearman_stage_vs_best_layer"] == pytest.approx(1.0)


def test_split_indices_are_disjoint_and_cover() -> None:
    generator = torch.Generator()
    generator.manual_seed(0)
    train_idx, test_idx = split_indices(21, generator)
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    assert sorted(train_idx.tolist() + test_idx.tolist()) == list(range(21))
