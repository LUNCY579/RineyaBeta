"""Cache pooled hidden states from the FROZEN backbone at arbitrary depths.

Why this module exists
----------------------
``scripts/permutation_preference.py`` consumed a cached tensor under the key
``SYM27`` that **no committed script produced**. The 24-permutation result in
``FINDINGS_G3_SEEDS.md`` section 2b -- the one that refuted the intrinsic
reverse-preference hypothesis -- was therefore not reproducible from a clean
checkout. This module closes that gap and generalises it: the checkpoint depths
are an argument, so the same code serves K=4 replication and K=8 extension.

Contract with the trained runs
------------------------------
A cache is scientifically comparable to the V1 runs only if it samples the same
tensor they did. Three things must match, and all three are recorded in the cache:

1. **Readout.** V1 used ``checkpoint_readout: final_norm``, which applies the
   model's final norm to ``hidden_states[L]`` for every ``L < num_hidden_layers``
   and skips it at the top, where HF has already applied it. Reproduced here via
   the same ``dast.model`` helpers rather than a private copy.
2. **Pooling.** Mean over PROMPT tokens only, via ``pool_prompt_hidden``. The
   states are predicted from the prompt alone; nothing from the response leaks in.
3. **Tokenization.** The same ``CausalBatchCollator``, so ``prompt_mask`` is
   identical to training's, including ``preserve_target`` truncation.

No LoRA is loaded and no parameter requires grad: this is the *untouched*
backbone, which is exactly the reference point section 2a needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from dast.alignment import resolve_checkpoint_layers
from dast.config import ARM_SPECS, DASTConfig
from dast.data import build_examples, load_records
from dast.model import infer_num_hidden_layers, locate_final_norm
from dast.pooling import pool_prompt_hidden
from dast.tokenization import CausalBatchCollator

CACHE_SCHEMA_VERSION = "dast-depth-cache-v1"


@dataclass(frozen=True)
class DepthCache:
    """Pooled frozen hidden states plus everything needed to trust them."""

    hidden: torch.Tensor  # [num_problems, num_checkpoints, hidden_size]
    layers: tuple[int, ...]  # one-indexed post-layer hidden-state indices
    problem_ids: tuple[str, ...]
    categories: tuple[str, ...]
    generators: tuple[str, ...]
    answer_types: tuple[str, ...]
    trajectory_lengths: tuple[int, ...]
    readout: str
    pooling: str
    num_hidden_layers: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3:
            raise ValueError(f"hidden must be rank 3, got {tuple(self.hidden.shape)}")
        count, stages, _ = self.hidden.shape
        if stages != len(self.layers):
            raise ValueError(
                f"hidden has {stages} checkpoints but {len(self.layers)} layers were recorded"
            )
        for name, values in (
            ("problem_ids", self.problem_ids),
            ("categories", self.categories),
            ("generators", self.generators),
            ("answer_types", self.answer_types),
            ("trajectory_lengths", self.trajectory_lengths),
        ):
            if len(values) != count:
                raise ValueError(f"{name} has {len(values)} entries for {count} problems")

    @property
    def num_problems(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def num_stages(self) -> int:
        return int(self.hidden.shape[1])

    def as_blob(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "hidden": self.hidden,
            "layers": list(self.layers),
            "problem_ids": list(self.problem_ids),
            "categories": list(self.categories),
            "generators": list(self.generators),
            "answer_types": list(self.answer_types),
            "trajectory_lengths": list(self.trajectory_lengths),
            "readout": self.readout,
            "pooling": self.pooling,
            "num_hidden_layers": self.num_hidden_layers,
            "metadata": self.metadata,
        }

    @classmethod
    def from_blob(cls, blob: dict[str, Any]) -> "DepthCache":
        version = blob.get("schema_version")
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"depth cache schema {version!r} is not {CACHE_SCHEMA_VERSION!r}; "
                "rebuild the cache rather than reinterpreting it"
            )
        return cls(
            hidden=blob["hidden"],
            layers=tuple(int(item) for item in blob["layers"]),
            problem_ids=tuple(blob["problem_ids"]),
            categories=tuple(blob["categories"]),
            generators=tuple(blob["generators"]),
            answer_types=tuple(blob["answer_types"]),
            trajectory_lengths=tuple(int(item) for item in blob["trajectory_lengths"]),
            readout=blob["readout"],
            pooling=blob["pooling"],
            num_hidden_layers=int(blob["num_hidden_layers"]),
            metadata=dict(blob.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.as_blob(), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DepthCache":
        return cls.from_blob(torch.load(Path(path), weights_only=False))


def uniform_checkpoint_fractions(num_checkpoints: int, *, top_fraction: float) -> tuple[float, ...]:
    """Evenly spaced fractions whose last entry is ``top_fraction``.

    ``top_fraction`` exists because a ``final_norm`` readout must NOT land on the
    top layer: ``hidden_states[num_hidden_layers]`` is already post-final-norm and
    ``DASTModel`` refuses it. V1 used 0.964 on a 28-layer model to land on 27.
    """

    if num_checkpoints < 1:
        raise ValueError("num_checkpoints must be positive")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    if num_checkpoints == 1:
        return (top_fraction,)
    step = top_fraction / num_checkpoints
    return tuple(round(step * (index + 1), 6) for index in range(num_checkpoints))


def resolve_cache_layers(
    num_hidden_layers: int,
    *,
    layers: Sequence[int] | None = None,
    num_checkpoints: int | None = None,
    readout: str = "final_norm",
) -> tuple[int, ...]:
    """Resolve and validate checkpoint depths, mirroring the V1 guards exactly."""

    if layers is None:
        if num_checkpoints is None:
            raise ValueError("pass either layers or num_checkpoints")
        # Keep the deepest checkpoint strictly below the top, as V1 did.
        top = 1.0 - 0.5 / num_hidden_layers - 1e-6
        layers = resolve_checkpoint_layers(
            num_hidden_layers,
            uniform_checkpoint_fractions(num_checkpoints, top_fraction=top),
        )
    resolved = tuple(int(item) for item in layers)
    if not resolved:
        raise ValueError("no checkpoint layers resolved")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate checkpoint layers: {resolved}")
    if sorted(resolved) != list(resolved):
        raise ValueError(f"checkpoint layers must be ascending: {resolved}")
    # Layer 0 is admitted deliberately. ``hidden_states[0]`` is the token-embedding
    # output, before any decoder layer. It is never a valid DAST supervision target,
    # but it is the indispensable CONTROL for the layer-by-stage profile: the frozen
    # semantic targets are themselves built from the input embedding
    # (`FrozenTextTargetEncoder`), so if stage decodability at layer 0 already
    # matches the best transformer layer, the profile is measuring lexical overlap
    # in embedding space rather than anything the depth stack computes.
    if min(resolved) < 0 or max(resolved) > num_hidden_layers:
        raise ValueError(f"layers {resolved} fall outside 0..{num_hidden_layers}")
    if readout == "final_norm" and max(resolved) >= num_hidden_layers:
        raise ValueError(
            f"readout='final_norm' cannot take layer {max(resolved)} of {num_hidden_layers}: "
            "the top hidden state is already post-final-norm and measures as a near-orthogonal "
            "outlier against the others. This mirrors the guard in DASTModel and keeps the cache "
            "comparable to the V1 runs."
        )
    return resolved


@torch.no_grad()
def build_depth_cache(
    base_model: torch.nn.Module,
    tokenizer: Any,
    config: DASTConfig,
    *,
    layers: Sequence[int] | None = None,
    num_checkpoints: int | None = None,
    split: str | None = None,
    limit: int | None = None,
    readout: str = "final_norm",
    batch_size: int = 2,
    device: torch.device | str = "cpu",
    progress: bool = True,
) -> DepthCache:
    """Run the frozen backbone once and pool prompt hidden states at ``layers``."""

    if readout not in ("raw", "final_norm"):
        raise ValueError(f"unsupported readout: {readout}")
    num_hidden_layers = infer_num_hidden_layers(base_model)
    resolved_layers = resolve_cache_layers(
        num_hidden_layers, layers=layers, num_checkpoints=num_checkpoints, readout=readout
    )

    final_norm = locate_final_norm(base_model) if readout == "final_norm" else None
    if readout == "final_norm" and final_norm is None:
        raise ValueError("readout='final_norm' requires a locatable final norm layer")

    split = split or config.data.eval_split
    records = load_records(
        config.data_path_for_split(split),
        split=split,
        limit=limit if limit is not None else config.data.max_eval_records,
    )
    if not records:
        raise ValueError(f"no records for split {split!r}")
    # The arm only affects which TARGET texts an example carries; the hidden side is
    # arm-independent. state_ordered is used so canonical_texts is populated.
    examples = build_examples(
        records,
        arm=ARM_SPECS["state_ordered"],
        num_stages=config.num_stages,
        seed=config.experiment.seed,
    )
    collator = CausalBatchCollator(
        tokenizer,
        system_prompt=config.data.system_prompt,
        max_length=config.model.max_length,
        truncation=config.data.truncation,
    )

    device = torch.device(device)
    base_model.eval()
    pooled_rows: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collator(chunk)
        outputs = base_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        prompt_mask = batch["prompt_mask"].to(device)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= max(resolved_layers):
            raise RuntimeError(
                f"model returned {0 if hidden_states is None else len(hidden_states)} hidden "
                f"states; need index {max(resolved_layers)}"
            )
        stages = []
        for layer_number in resolved_layers:
            state = hidden_states[layer_number]
            if final_norm is not None and layer_number < num_hidden_layers:
                state = final_norm(state)
            stages.append(
                pool_prompt_hidden(state, prompt_mask, config.model.pooling).float().cpu()
            )
        pooled_rows.append(torch.stack(stages, dim=1))
        if progress and (start // batch_size) % 25 == 0:
            print(f"  pooled {min(start + batch_size, len(examples))}/{len(examples)}", flush=True)

    hidden = torch.cat(pooled_rows, dim=0)
    return DepthCache(
        hidden=hidden,
        layers=resolved_layers,
        problem_ids=tuple(item.record.problem_id for item in examples),
        categories=tuple(item.record.category for item in examples),
        generators=tuple(getattr(item.record, "generator", "") or "" for item in examples),
        answer_types=tuple(item.record.answer_type for item in examples),
        trajectory_lengths=tuple(len(item.trajectory.canonical_texts) for item in examples),
        readout=readout,
        pooling=config.model.pooling,
        num_hidden_layers=num_hidden_layers,
        metadata={
            "split": split,
            "num_problems": int(hidden.shape[0]),
            "hidden_size": int(hidden.shape[-1]),
            "model": config.model.name_or_path,
            "max_length": config.model.max_length,
            "truncation": config.data.truncation,
            "system_prompt": config.data.system_prompt,
            "config_sha256": config.canonical_sha256(),
            "seed": config.experiment.seed,
            "lora_loaded": False,
        },
    )
