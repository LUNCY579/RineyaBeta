"""Map variable-length semantic trajectories onto fixed Transformer depths."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from dast.config import ArmSpec


@dataclass(frozen=True)
class AlignedTrajectory:
    training_texts: tuple[str, ...]
    canonical_texts: tuple[str, ...]
    canonical_source_indices: tuple[int, ...]
    training_source_indices: tuple[int, ...]
    assignment: tuple[int, ...]
    repeated_stage_mask: tuple[bool, ...]


def resolve_checkpoint_layers(num_hidden_layers: int, fractions: Sequence[float]) -> tuple[int, ...]:
    """Resolve fractions to one-indexed post-layer hidden-state checkpoints.

    Hugging Face returns ``hidden_states[0]`` for embeddings and
    ``hidden_states[i]`` after layer ``i``.  A 28-layer model with fractions
    ``(.25, .5, .75, 1.0)`` therefore resolves to ``(7, 14, 21, 28)``.
    """

    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be positive")
    resolved: list[int] = []
    for fraction in fractions:
        layer = max(1, min(num_hidden_layers, int(round(num_hidden_layers * fraction))))
        if resolved and layer <= resolved[-1]:
            layer = resolved[-1] + 1
        if layer > num_hidden_layers:
            raise ValueError(
                "checkpoint fractions collapse to duplicate layers; reduce the number of stages "
                f"for a {num_hidden_layers}-layer model"
            )
        resolved.append(layer)
    return tuple(resolved)


def _progress_indices(num_targets: int, num_stages: int) -> tuple[int, ...]:
    if num_targets < 1:
        raise ValueError("a semantic trajectory must contain at least one target")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    if num_stages == 1:
        return (num_targets - 1,)
    # Pin both endpoints and distribute intermediate checkpoints by normalized
    # progress.  This intentionally permits repeats when a task has fewer
    # semantic transitions than depth checkpoints.
    return tuple(round(stage * (num_targets - 1) / (num_stages - 1)) for stage in range(num_stages))


def _deterministic_permutation(length: int, problem_id: str, seed: int) -> tuple[int, ...]:
    if length <= 1:
        return tuple(range(length))
    digest = hashlib.sha256(f"{seed}:{problem_id}:dast-depth-shuffle".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(length))
    rng.shuffle(order)
    # The negative control must not silently become the ordered arm for a sample.
    if order == list(range(length)):
        order = order[1:] + order[:1]
    return tuple(order)


def align_trajectory(
    raw_targets: Sequence[str],
    *,
    num_stages: int,
    arm: ArmSpec,
    problem_id: str,
    seed: int,
) -> AlignedTrajectory:
    """Return matched canonical and arm-specific depth targets.

    Every semantic control receives the same target multiset and number of losses.
    Only the depth assignment changes.  ``final_repeated`` is the explicit generic
    deep-supervision control and therefore changes the multiset by design.
    """

    cleaned = tuple(str(item).strip() for item in raw_targets if str(item).strip())
    if not cleaned:
        raise ValueError(f"problem {problem_id!r} has no usable trajectory targets")
    source_indices = _progress_indices(len(cleaned), num_stages)
    canonical = tuple(cleaned[index] for index in source_indices)

    if arm.assignment == "ordered":
        assignment = tuple(range(num_stages))
    elif arm.assignment == "reverse":
        assignment = tuple(reversed(range(num_stages)))
    elif arm.assignment == "shuffled":
        assignment = _deterministic_permutation(num_stages, problem_id, seed)
    elif arm.assignment == "final_repeated":
        assignment = tuple(num_stages - 1 for _ in range(num_stages))
    elif arm.assignment == "none":
        return AlignedTrajectory((), canonical, source_indices, (), (), ())
    else:  # pragma: no cover - ArmSpec is validated centrally
        raise ValueError(f"unsupported assignment: {arm.assignment}")

    training = tuple(canonical[index] for index in assignment)
    training_source_indices = tuple(source_indices[index] for index in assignment)
    repeated = tuple(index > 0 and training_source_indices[index] == training_source_indices[index - 1] for index in range(num_stages))
    return AlignedTrajectory(
        training_texts=training,
        canonical_texts=canonical,
        canonical_source_indices=source_indices,
        training_source_indices=training_source_indices,
        assignment=assignment,
        repeated_stage_mask=repeated,
    )
