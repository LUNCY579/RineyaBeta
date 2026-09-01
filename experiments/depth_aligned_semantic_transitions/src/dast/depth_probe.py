"""Frozen-backbone ridge probes: what state-to-depth structure is already there?

This is measurement, not training. Every function here fits a closed-form ridge
readout on held-out splits of the UNTOUCHED backbone and reports how well
semantic stages are decodable at each depth. Nothing here licenses a causal claim:
decodability is not computation, and `FINDINGS_G3_SEEDS.md` section 2a is the
standing reminder that a probe beating the trained projector says the information
was already present, not that anything reasons.

Two questions, two estimators
-----------------------------
`permutation_landscape` asks: given K fixed depths and K semantic stages, does the
canonical forward assignment fit better than the alternatives? This replicates and
generalises the section 2b study.

`layer_stage_profile` asks the sharper question: across ALL usable layers, which
depth best hosts each semantic stage? A chronological depth axis predicts a
diagonal ridge -- early stages peaking early, late stages peaking late. This does
not presuppose K depths or an assignment at all.

The shared readout
------------------
Both use ONE linear map from hidden state to semantic state, applied at every
depth -- the estimator analogue of `SharedStateProjector`, which V1 also shares
across depth. A per-depth readout would trivially fit any assignment and answer a
different question.

Efficiency note (why exhaustive permutation search is cheap)
-----------------------------------------------------------
For a permutation pi, the ridge normal equations are ``A W = X^T Y(pi)`` where
``A = X^T X + lam I`` does NOT depend on pi -- only the right-hand side does. And
that side decomposes: ``X^T Y(pi) = sum_k M[k, pi(k)]`` with
``M[k, j] = sum_n x_{n,k} (x) s_{n,j}``. So a single factorisation of ``A`` plus
``K^2`` precomputed blocks turns each permutation into a handful of adds. That is
what makes 6! x 20 splits a few seconds instead of hours.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

DEFAULT_RIDGE = 1.0


def center_targets(targets: torch.Tensor) -> torch.Tensor:
    """Apply V1's ``target_centering='within_problem'`` exactly.

    Subtract each problem's own mean over its CANONICAL stages, then renormalise.
    Matches `dast.training._encode_targets`. The consequence worth stating: K
    centred vectors span at most K-1 dimensions, so the centred target set is
    rank-deficient by construction and no readout can separate all K directions
    independently.
    """

    if targets.ndim != 3:
        raise ValueError(f"targets must be rank 3 [problem, stage, dim], got {tuple(targets.shape)}")
    return F.normalize(targets - targets.mean(dim=1, keepdim=True), dim=-1)


def _design(hidden: torch.Tensor) -> torch.Tensor:
    """Flatten [n, k, d_hidden] to [(n k), d_hidden + 1] with a bias column."""

    count, stages, width = hidden.shape
    flat = hidden.reshape(count * stages, width)
    return torch.cat([flat, flat.new_ones(count * stages, 1)], dim=1)


@dataclass(frozen=True)
class ProbeScore:
    l_state: float
    diagonal_cosine: float
    order_margin: float
    top1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "L_state": self.l_state,
            "diagonal_cosine": self.diagonal_cosine,
            "order_margin": self.order_margin,
            "top1": self.top1,
        }


def split_indices(count: int, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random half/half split. A deterministic split is used when generator is None."""

    if count < 4:
        raise ValueError("need at least 4 problems to split")
    half = count // 2
    order = torch.arange(count) if generator is None else torch.randperm(count, generator=generator)
    return order[:half], order[half:]


def _solve_blocks(
    hidden_train: torch.Tensor,
    targets_train: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cholesky factor of A, block tensor M) for the factorised solve.

    ``M[k, j]`` is the right-hand-side contribution of putting canonical stage
    ``j`` at depth ``k``. Shape ``[K, K, d_hidden + 1, d_state]``.
    """

    stages = hidden_train.shape[1]
    design = _design(hidden_train)
    width = design.shape[1]
    gram = design.T @ design + ridge * torch.eye(width, dtype=design.dtype)
    factor = torch.linalg.cholesky(gram)

    # x_{n,k} for a fixed k, over all n.
    per_stage = design.reshape(hidden_train.shape[0], stages, width)
    # M[k, j] = sum_n x_{n,k} (x) s_{n,j}
    blocks = torch.einsum("nkw,njd->kjwd", per_stage, targets_train)
    return factor, blocks


def _score_permutations(
    factor: torch.Tensor,
    blocks: torch.Tensor,
    hidden_test: torch.Tensor,
    targets_test: torch.Tensor,
    permutations: Sequence[Sequence[int]],
    *,
    batch: int = 64,
) -> list[ProbeScore]:
    """Score many permutations sharing one factorisation.

    Metrics are identical in definition to `dast.losses.depth_alignment_metrics`
    and to the original section 2b script, but computed against UNPERMUTED targets
    and re-indexed, which is algebraically the same and avoids materialising a
    permuted target grid per permutation.
    """

    stages = hidden_test.shape[1]
    count_test = hidden_test.shape[0]
    state_dim = targets_test.shape[-1]
    width = blocks.shape[2]

    # G[k, j] = A^-1 M[k, j]; one batched triangular solve for all K^2 blocks.
    rhs = blocks.permute(2, 0, 1, 3).reshape(width, stages * stages * state_dim)
    solved = torch.cholesky_solve(rhs, factor)
    gain = solved.reshape(width, stages, stages, state_dim).permute(1, 2, 0, 3)

    # projected[k, kp, j] = x_test[:, k] @ G[kp, j]  ->  [K, K, K, n_test, d_state]
    design_test = _design(hidden_test)
    flat_gain = gain.reshape(stages * stages, width, state_dim).permute(1, 0, 2).reshape(width, -1)
    projected = (design_test @ flat_gain).reshape(
        count_test, stages, stages, stages, state_dim
    )
    # dims: [n_test, k(depth of the hidden state), kp, j] -> reorder to [k, kp, j, n, d]
    projected = projected.permute(1, 2, 3, 0, 4).contiguous()

    targets_unit = F.normalize(targets_test, dim=-1)
    arange = torch.arange(stages)
    results: list[ProbeScore] = []

    for start in range(0, len(permutations), batch):
        chunk = permutations[start : start + batch]
        index = torch.tensor([list(perm) for perm in chunk], dtype=torch.long)  # [B, K]
        size = index.shape[0]
        # predictions[b, n, k] = sum_kp projected[k, kp, perm_b(kp), n]
        predictions = torch.zeros(
            size, count_test, stages, state_dim, dtype=projected.dtype
        )
        for prior in range(stages):
            # projected[:, prior] -> [K(k), K(j), n, d]; pick j = perm_b(prior)
            picked = projected[:, prior][:, index[:, prior]]  # [K(k), B, n, d]
            predictions += picked.permute(1, 2, 0, 3)
        predictions = F.normalize(predictions, dim=-1)

        # raw[b, n, k, j] = <pred[b, n, k], target[n, j]> against CANONICAL j
        raw = torch.einsum("bnkd,njd->bnkj", predictions, targets_unit)
        # assigned[b, n, k, 0] = perm_b(k): the canonical stage depth k was given.
        assigned = index.view(size, 1, stages, 1).expand(size, count_test, stages, 1)
        diagonal = raw.gather(3, assigned).squeeze(3)  # [B, n, K]
        off_mean = (raw.sum(dim=3) - diagonal) / max(stages - 1, 1)
        best = raw.argmax(dim=3)  # [B, n, K] canonical index that fits best
        target_index = index.unsqueeze(1).expand(size, count_test, stages)

        for row in range(size):
            diag_mean = float(diagonal[row].mean())
            results.append(
                ProbeScore(
                    l_state=1.0 - diag_mean,
                    diagonal_cosine=diag_mean,
                    order_margin=diag_mean - float(off_mean[row].mean()),
                    top1=float(best[row].eq(target_index[row]).float().mean()),
                )
            )
    return results


def per_problem_permutation_losses(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    fit_index: torch.Tensor,
    eval_index: torch.Tensor,
    permutations: Sequence[Sequence[int]],
    *,
    ridge: float = DEFAULT_RIDGE,
    batch: int = 64,
) -> torch.Tensor:
    """Per-problem `L_state` under every permutation. Returns [n_eval, n_perms].

    The readout is fitted ONLY on ``fit_index``, so an evaluated problem never
    contributes to the map that scores it. This is the estimator C2 needs: a
    per-problem preference computed by fitting on the evaluated item would
    manufacture heterogeneity out of overfitting.
    """

    hidden = hidden.double()
    targets = targets.double()
    stages = hidden.shape[1]
    factor, blocks = _solve_blocks(hidden[fit_index], targets[fit_index], ridge)

    hidden_eval = hidden[eval_index]
    targets_eval = targets[eval_index]
    count = hidden_eval.shape[0]
    state_dim = targets_eval.shape[-1]
    width = blocks.shape[2]

    rhs = blocks.permute(2, 0, 1, 3).reshape(width, stages * stages * state_dim)
    gain = torch.cholesky_solve(rhs, factor).reshape(
        width, stages, stages, state_dim
    ).permute(1, 2, 0, 3)
    design = _design(hidden_eval)
    flat_gain = gain.reshape(stages * stages, width, state_dim).permute(1, 0, 2).reshape(width, -1)
    projected = (design @ flat_gain).reshape(
        count, stages, stages, stages, state_dim
    ).permute(1, 2, 3, 0, 4).contiguous()

    targets_unit = F.normalize(targets_eval, dim=-1)
    losses = torch.empty(count, len(permutations), dtype=torch.float64)
    for start in range(0, len(permutations), batch):
        chunk = permutations[start : start + batch]
        index = torch.tensor([list(perm) for perm in chunk], dtype=torch.long)
        size = index.shape[0]
        predictions = torch.zeros(size, count, stages, state_dim, dtype=projected.dtype)
        for prior in range(stages):
            predictions += projected[:, prior][:, index[:, prior]].permute(1, 2, 0, 3)
        predictions = F.normalize(predictions, dim=-1)
        raw = torch.einsum("bnkd,njd->bnkj", predictions, targets_unit)
        assigned = index.view(size, 1, stages, 1).expand(size, count, stages, 1)
        diagonal = raw.gather(3, assigned).squeeze(3)  # [B, n, K]
        losses[:, start : start + size] = (1.0 - diagonal.mean(dim=2)).T
    return losses


def kendall_tau(perm: Sequence[int]) -> float:
    """Kendall tau of a permutation against the identity, in [-1, 1]."""

    stages = len(perm)
    if stages < 2:
        return 1.0
    concordant = sum(
        1 if perm[i] < perm[j] else -1 for i in range(stages) for j in range(i + 1, stages)
    )
    return concordant / (stages * (stages - 1) / 2)


def structured_permutations(stages: int) -> dict[str, tuple[int, ...]]:
    """Named permutations worth reporting whether or not the search is exhaustive."""

    identity = tuple(range(stages))
    named: dict[str, tuple[int, ...]] = {
        "identity": identity,
        "reverse": tuple(reversed(identity)),
    }
    if stages >= 4:
        first_half = identity[: stages // 2]
        second_half = identity[stages // 2 :]
        named["block_reversal"] = (*reversed(first_half), *reversed(second_half))
        named["half_swap"] = (*second_half, *first_half)
        named["odd_even_interleave"] = (*identity[::2], *identity[1::2])
        # center-out: middle stages read first, endpoints last.
        centre = sorted(identity, key=lambda index: abs(index - (stages - 1) / 2))
        named["center_out"] = tuple(centre)
        named["outside_in"] = tuple(reversed(centre))
    if stages >= 2:
        adjacent = list(identity)
        adjacent[0], adjacent[1] = adjacent[1], adjacent[0]
        named["swap_first_two"] = tuple(adjacent)
        adjacent = list(identity)
        adjacent[-1], adjacent[-2] = adjacent[-2], adjacent[-1]
        named["swap_last_two"] = tuple(adjacent)
        endpoints = list(identity)
        endpoints[0], endpoints[-1] = endpoints[-1], endpoints[0]
        named["swap_endpoints"] = tuple(endpoints)
    return named


def permutation_landscape(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    permutations: Iterable[Sequence[int]] | None = None,
    splits: int = 20,
    ridge: float = DEFAULT_RIDGE,
    seed: int = 1701,
) -> dict:
    """Rank every requested state-to-depth assignment across many random splits.

    ``hidden``  [problems, K, hidden_size] frozen pooled states.
    ``targets`` [problems, K, state_dim] canonical targets, ALREADY centred.
    """

    if hidden.shape[0] != targets.shape[0]:
        raise ValueError("hidden and targets disagree on problem count")
    if hidden.shape[1] != targets.shape[1]:
        raise ValueError("hidden and targets disagree on stage count")
    stages = hidden.shape[1]
    perms = (
        [tuple(range(stages))] if stages < 2 else
        list(permutations) if permutations is not None
        else list(itertools.permutations(range(stages)))
    )
    perms = [tuple(int(x) for x in perm) for perm in perms]
    named = structured_permutations(stages)
    label_of = {perm: name for name, perm in named.items()}

    hidden = hidden.double()
    targets = targets.double()
    per_split: list[list[ProbeScore]] = []
    generator = torch.Generator()
    for index in range(splits):
        generator.manual_seed(seed + index)
        train_idx, test_idx = split_indices(hidden.shape[0], generator)
        factor, blocks = _solve_blocks(hidden[train_idx], targets[train_idx], ridge)
        per_split.append(
            _score_permutations(factor, blocks, hidden[test_idx], targets[test_idx], perms)
        )

    rows = []
    for position, perm in enumerate(perms):
        losses = [split[position].l_state for split in per_split]
        ranks = []
        for split in per_split:
            order = sorted(range(len(perms)), key=lambda p: split[p].l_state)
            ranks.append(order.index(position) + 1)
        rows.append(
            {
                "perm": list(perm),
                "name": label_of.get(perm, ""),
                "kendall_tau_vs_identity": kendall_tau(perm),
                "L_state_mean": sum(losses) / len(losses),
                "L_state_min": min(losses),
                "L_state_max": max(losses),
                "rank_median": sorted(ranks)[len(ranks) // 2],
                "rank_best": min(ranks),
                "rank_worst": max(ranks),
                "times_ranked_first": sum(1 for r in ranks if r == 1),
                "order_margin_mean": sum(s.order_margin for s in (sp[position] for sp in per_split))
                / len(per_split),
                "top1_mean": sum(s.top1 for s in (sp[position] for sp in per_split)) / len(per_split),
            }
        )
    rows.sort(key=lambda row: row["L_state_mean"])

    spreads = [
        max(split[p].l_state for p in range(len(perms)))
        - min(split[p].l_state for p in range(len(perms)))
        for split in per_split
    ]
    identity_row = next(row for row in rows if tuple(row["perm"]) == tuple(range(stages)))
    reverse_row = next(
        (row for row in rows if tuple(row["perm"]) == tuple(reversed(range(stages)))), None
    )
    return {
        "num_problems": int(hidden.shape[0]),
        "num_stages": stages,
        "num_permutations": len(perms),
        "exhaustive": len(perms) == math.factorial(stages),
        "splits": splits,
        "ridge": ridge,
        "permutations_best_first": rows,
        "summary": {
            "identity_rank_median": identity_row["rank_median"],
            "identity_L_state_mean": identity_row["L_state_mean"],
            "identity_times_first": identity_row["times_ranked_first"],
            "reverse_rank_median": reverse_row["rank_median"] if reverse_row else None,
            "reverse_L_state_mean": reverse_row["L_state_mean"] if reverse_row else None,
            "reverse_times_first": reverse_row["times_ranked_first"] if reverse_row else None,
            "best_perm": rows[0]["perm"],
            "best_name": rows[0]["name"],
            "worst_perm": rows[-1]["perm"],
            "L_state_spread_mean": sum(spreads) / len(spreads),
            "L_state_spread_as_fraction_of_L_state": (sum(spreads) / len(spreads))
            / identity_row["L_state_mean"],
        },
    }


def layer_stage_profile(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    layers: Sequence[int],
    *,
    splits: int = 10,
    ridge: float = DEFAULT_RIDGE,
    seed: int = 1701,
) -> dict:
    """For every layer and every semantic stage, how well is that stage decodable?

    ``hidden`` is [problems, num_layers, hidden_size] pooled at ALL candidate
    depths; ``targets`` is [problems, K, state_dim], already centred. A separate
    readout is fitted per (layer, stage) pair -- here that is the right estimator,
    because the question is about each cell independently rather than about one
    interface shared across depth.

    A chronological depth axis predicts the argmax layer to increase with stage.
    """

    if hidden.shape[0] != targets.shape[0]:
        raise ValueError("hidden and targets disagree on problem count")
    if hidden.shape[1] != len(layers):
        raise ValueError("hidden layer count does not match the layers argument")
    num_layers = len(layers)
    stages = targets.shape[1]
    state_dim = targets.shape[2]
    hidden = hidden.double()
    targets = targets.double()

    cosine = torch.zeros(splits, num_layers, stages, dtype=torch.float64)
    generator = torch.Generator()
    for split in range(splits):
        generator.manual_seed(seed + split)
        train_idx, test_idx = split_indices(hidden.shape[0], generator)
        for layer_position in range(num_layers):
            train_hidden = hidden[train_idx, layer_position]
            design = torch.cat([train_hidden, train_hidden.new_ones(len(train_idx), 1)], dim=1)
            gram = design.T @ design + ridge * torch.eye(design.shape[1], dtype=design.dtype)
            factor = torch.linalg.cholesky(gram)
            # One readout per (layer, stage): stack the K stage targets side by side
            # so a single solve yields all of them, then split back out.
            stacked = targets[train_idx].reshape(len(train_idx), stages * state_dim)
            weights = torch.cholesky_solve(design.T @ stacked, factor)
            weights = weights.reshape(design.shape[1], stages, state_dim)
            test_hidden = hidden[test_idx, layer_position]
            test_design = torch.cat(
                [test_hidden, test_hidden.new_ones(len(test_idx), 1)], dim=1
            )
            for stage in range(stages):
                predicted = F.normalize(test_design @ weights[:, stage], dim=-1)
                actual = F.normalize(targets[test_idx, stage], dim=-1)
                cosine[split, layer_position, stage] = (predicted * actual).sum(dim=-1).mean()

    mean = cosine.mean(dim=0)
    argmax_layer = [int(mean[:, stage].argmax()) for stage in range(stages)]
    return {
        "num_problems": int(hidden.shape[0]),
        "layers": list(int(layer) for layer in layers),
        "num_stages": stages,
        "splits": splits,
        "ridge": ridge,
        "cosine_mean": mean.tolist(),
        "cosine_std": cosine.std(dim=0).tolist(),
        "best_layer_per_stage": [layers[position] for position in argmax_layer],
        "best_layer_position_per_stage": argmax_layer,
        "monotone_in_stage": all(
            argmax_layer[stage] <= argmax_layer[stage + 1] for stage in range(stages - 1)
        ),
        "spearman_stage_vs_best_layer": _spearman(
            list(range(stages)), [float(position) for position in argmax_layer]
        ),
    }


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman correlation. Returns 0.0 when either side has no variance."""

    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            shared = (position + end) / 2 + 1
            for index in range(position, end + 1):
                ranks[order[index]] = shared
            position = end + 1
        return ranks

    first_ranks, second_ranks = rank(first), rank(second)
    count = len(first_ranks)
    mean_first = sum(first_ranks) / count
    mean_second = sum(second_ranks) / count
    numerator = sum(
        (first_ranks[i] - mean_first) * (second_ranks[i] - mean_second) for i in range(count)
    )
    denominator = math.sqrt(
        sum((value - mean_first) ** 2 for value in first_ranks)
        * sum((value - mean_second) ** 2 for value in second_ranks)
    )
    return numerator / denominator if denominator > 0 else 0.0
