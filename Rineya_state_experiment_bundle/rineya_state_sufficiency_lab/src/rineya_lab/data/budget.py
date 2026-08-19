"""Exposure and token accounting, and the budget-matching controls.

The SPLIT dataset expands one source problem into ~3.96 rows. If SPLIT simply
runs longer, any measured difference confounds *representation* with *quantity*.
The protocol therefore requires reporting more than one budget regime:

native                 each representation runs its own natural epoch
matched_updates        identical number of optimizer updates
matched_target_tokens  identical cumulative supervised (target) token count
matched_problems       identical unique source-problem exposure per epoch

``BudgetGuard`` enforces whichever limits a config sets and records the reason a
run stopped, so a truncated run is never mistaken for a completed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rineya_lab.config import BudgetConfig, LabConfig
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("data.budget")


@dataclass
class ExposureAccount:
    """Running totals reported in every run manifest."""

    records_seen: int = 0
    unique_problems: set[str] = field(default_factory=set)
    input_tokens: int = 0
    target_tokens: int = 0
    padded_tokens: int = 0
    supervised_positions: int = 0
    optimizer_updates: int = 0
    micro_batches: int = 0
    truncated_prompts: int = 0
    truncated_targets: int = 0
    dropped_records: int = 0
    weighted_records: float = 0.0

    def observe_example(
        self,
        *,
        problem_id: str,
        input_tokens: int,
        target_tokens: int,
        weight: float = 1.0,
        prompt_truncated: bool = False,
        target_truncated: bool = False,
    ) -> None:
        self.records_seen += 1
        self.unique_problems.add(problem_id)
        self.input_tokens += input_tokens
        self.target_tokens += target_tokens
        self.supervised_positions += target_tokens
        self.weighted_records += weight
        if prompt_truncated:
            self.truncated_prompts += 1
        if target_truncated:
            self.truncated_targets += 1

    def observe_batch(self, *, padded_tokens: int) -> None:
        self.micro_batches += 1
        self.padded_tokens += padded_tokens

    def observe_update(self) -> None:
        self.optimizer_updates += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.target_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "unique_problems_seen": len(self.unique_problems),
            "input_tokens": self.input_tokens,
            "target_tokens": self.target_tokens,
            "total_tokens": self.total_tokens,
            "padded_tokens": self.padded_tokens,
            "supervised_positions": self.supervised_positions,
            "optimizer_updates": self.optimizer_updates,
            "micro_batches": self.micro_batches,
            "truncated_prompts": self.truncated_prompts,
            "truncated_targets": self.truncated_targets,
            "dropped_records": self.dropped_records,
            "weighted_records": round(self.weighted_records, 6),
        }


@dataclass
class BudgetGuard:
    """Decides when a training run has consumed its allotted budget."""

    cfg: BudgetConfig
    stop_reason: str | None = None

    def limits(self) -> dict[str, int | None]:
        return {
            "max_optimizer_updates": self.cfg.max_optimizer_updates,
            "max_target_tokens": self.cfg.max_target_tokens,
            "max_total_tokens": self.cfg.max_total_tokens,
            "max_unique_problems": self.cfg.max_unique_problems,
            "max_records": self.cfg.max_records,
        }

    def should_stop(self, account: ExposureAccount) -> bool:
        checks = (
            ("max_optimizer_updates", account.optimizer_updates, self.cfg.max_optimizer_updates),
            ("max_target_tokens", account.target_tokens, self.cfg.max_target_tokens),
            ("max_total_tokens", account.total_tokens, self.cfg.max_total_tokens),
            ("max_unique_problems", len(account.unique_problems), self.cfg.max_unique_problems),
            ("max_records", account.records_seen, self.cfg.max_records),
        )
        for name, actual, limit in checks:
            if limit is not None and actual >= limit:
                if self.stop_reason is None:
                    self.stop_reason = f"{name} reached ({actual} >= {limit})"
                    LOGGER.info("Budget guard: %s", self.stop_reason)
                return True
        return False

    def accepts_more_problems(self, account: ExposureAccount, problem_id: str) -> bool:
        """Whether a record may be consumed under a unique-problem cap."""
        limit = self.cfg.max_unique_problems
        if limit is None or problem_id in account.unique_problems:
            return True
        return len(account.unique_problems) < limit


def measure_view(
    dataset,
    cfg: LabConfig,
    tokenizer,
    *,
    sample_limit: int | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Tokenize (a sample of) a training view and report its token budget.

    Used by ``rineya-lab budget`` to compute the matched-token stopping points
    *before* launching two GPU runs, which is the only practical way to make
    Regime T exact.
    """
    from rineya_lab.data.views import build_view_plan, iter_examples
    from rineya_lab.training.chat import ChatFormatter

    formatter = ChatFormatter(tokenizer, cfg.model)
    plan = build_view_plan(dataset, cfg, split=split)
    account = ExposureAccount()

    for count, example in enumerate(iter_examples(dataset, plan, cfg)):
        if sample_limit is not None and count >= sample_limit:
            break
        encoded = formatter.encode_example(
            example.messages,
            max_seq_len=cfg.training.max_seq_len,
            prompt_head_keep=cfg.training.prompt_head_keep,
        )
        account.observe_example(
            problem_id=example.problem_id,
            input_tokens=encoded.prompt_token_count,
            target_tokens=encoded.target_token_count,
            weight=example.weight,
            prompt_truncated=encoded.prompt_truncated,
            target_truncated=encoded.target_truncated,
        )

    measured = account.as_dict()
    sampled = sample_limit is not None and account.records_seen >= sample_limit
    planned_rows = plan.total_rows
    scale = planned_rows / account.records_seen if account.records_seen and sampled else 1.0
    return {
        "plan": plan.summary(),
        "measured": measured,
        "sampled": sampled,
        "projection_scale": round(scale, 6),
        "projected_target_tokens": int(round(measured["target_tokens"] * scale)),
        "projected_input_tokens": int(round(measured["input_tokens"] * scale)),
        "projected_total_tokens": int(round(measured["total_tokens"] * scale)),
        "projected_records": planned_rows,
        "effective_batch_size": cfg.training.per_device_batch_size * cfg.training.gradient_accumulation_steps,
        "projected_optimizer_updates": int(
            planned_rows // (cfg.training.per_device_batch_size * cfg.training.gradient_accumulation_steps)
        ),
    }


def matched_budget_recommendation(full: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    """Given two ``measure_view`` reports, propose matched stopping points.

    Every matched budget is the *minimum* of the two arms: raising one arm to
    meet the other would require data it does not have.
    """
    def pick(key: str) -> int:
        return int(min(full[key], split[key]))

    return {
        "matched_target_tokens": pick("projected_target_tokens"),
        "matched_total_tokens": pick("projected_total_tokens"),
        "matched_optimizer_updates": pick("projected_optimizer_updates"),
        "matched_records": pick("projected_records"),
        "matched_unique_problems": int(
            min(full["plan"]["unique_problems"], split["plan"]["unique_problems"])
        ),
        "full_arm": {k: full[k] for k in ("projected_target_tokens", "projected_records", "projected_optimizer_updates")},
        "split_arm": {k: split[k] for k in ("projected_target_tokens", "projected_records", "projected_optimizer_updates")},
        "note": (
            "Set budget.mode and the corresponding budget.max_* on BOTH arms. "
            "Matching one axis necessarily unmatches the others; report which axis was matched."
        ),
    }
