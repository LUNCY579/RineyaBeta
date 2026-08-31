#!/usr/bin/env python3
"""Deterministic, token-budgeted batch plan.

The plan is the single source of truth for what the optimizer sees. It is a pure
function of (row lengths, seed, epoch, budget), so it can be regenerated on resume
instead of being serialised and trusted -- and its hash detects any drift.

Why a token budget instead of a fixed batch size
------------------------------------------------
The healing mix spans 99-token chat rows and 2,577-token long-context rows. A
fixed batch size is wrong at both ends: it either wastes ~95% of every position on
padding for the short rows, or it OOMs on the long ones. Budgeting on
`rows x longest_row_in_batch` (the real padded cost) makes peak activation memory
roughly constant per micro-batch regardless of which corpus a row came from.

Why length grouping is safe here
--------------------------------
Rows are shuffled, cut into megabatches, sorted by length *within* a megabatch,
packed into micro-batches, and then the micro-batch order is shuffled again. So
padding stays low while the optimizer still sees a well-mixed sequence of batches.
Every step is seeded, so the result is reproducible.

Width bucketing: available, and measured not to be worth it
-----------------------------------------------------------
`width_bucket` rounds padded width up to a multiple of N, capping the number of
distinct batch shapes at `max_len / N`. The theory was that fewer shapes means
less allocator churn and therefore a tighter pre-flight probe.

Measured on B0, and it does not pay:

  bucket=1 (off)   padding 99.2%   probe 3,048   actual 3,880 MiB   2.88 h
  bucket=64        padding 81.6%   probe 3,466   actual 3,880 MiB   3.24 h

Actual peak is identical either way -- 3,880 MiB is the real working set, not a
function of shape diversity. Bucketing only narrowed the probe's under-estimate
(27% -> 12%) while costing 22% more padded positions and 12% wall clock. Kept as
an option, defaulted off.

The operational conclusion is that the probe is a fast *lower bound*, not a
predictor. Validate a new config with a short real run and read peak_reserved_mib
from train_log.jsonl; that takes two minutes and is authoritative.

Loss normalisation
------------------
Each step records `supervised_tokens`, the exact number of positions that
contribute to the loss across its micro-batches. A micro-batch scales its summed
token loss by `1 / step.supervised_tokens`, so accumulated gradients equal a true
token-mean over the whole optimizer step. That makes the loss independent of how
rows happened to be packed -- which a naive per-micro-batch mean is not.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MicroBatch:
    indices: tuple[int, ...]
    max_len: int            # longest row, i.e. the padded width
    supervised: int         # target tokens contributing to the loss
    actual: int             # real tokens present, before padding

    @property
    def padded_cost(self) -> int:
        return len(self.indices) * self.max_len

    @property
    def padding_efficiency(self) -> float:
        return self.actual / max(self.padded_cost, 1)


@dataclass(frozen=True)
class Step:
    micro: tuple[MicroBatch, ...]

    @property
    def supervised_tokens(self) -> int:
        return sum(m.supervised for m in self.micro)

    @property
    def rows(self) -> int:
        return sum(len(m.indices) for m in self.micro)

    @property
    def peak_padded_cost(self) -> int:
        return max(m.padded_cost for m in self.micro)

    @property
    def actual_tokens(self) -> int:
        return sum(m.actual for m in self.micro)


@dataclass
class Plan:
    steps: list[Step]
    epoch: int
    config: dict

    @property
    def total_rows(self) -> int:
        return sum(s.rows for s in self.steps)

    @property
    def peak_padded_cost(self) -> int:
        return max((s.peak_padded_cost for s in self.steps), default=0)

    @property
    def widest_micro_batch(self) -> MicroBatch | None:
        best = None
        for step in self.steps:
            for m in step.micro:
                if best is None or m.padded_cost > best.padded_cost:
                    best = m
        return best

    def hash(self) -> str:
        """Stable digest of the exact batching decisions."""
        h = hashlib.sha256()
        h.update(json.dumps(self.config, sort_keys=True).encode())
        h.update(f"|epoch={self.epoch}|".encode())
        for step in self.steps:
            for m in step.micro:
                h.update(b"(")
                h.update(",".join(str(i) for i in m.indices).encode())
                h.update(f"|{m.max_len}|{m.supervised})".encode())
            h.update(b";")
        return h.hexdigest()

    def summary(self) -> dict:
        widest = self.widest_micro_batch
        return {
            "epoch": self.epoch,
            "steps": len(self.steps),
            "rows": self.total_rows,
            "supervised_tokens": sum(s.supervised_tokens for s in self.steps),
            "actual_tokens": sum(s.actual_tokens for s in self.steps),
            "padded_positions": sum(m.padded_cost for s in self.steps for m in s.micro),
            "peak_padded_cost": self.peak_padded_cost,
            "widest_micro_batch": None if widest is None else {
                "rows": len(widest.indices), "max_len": widest.max_len,
                "padded_cost": widest.padded_cost},
            "hash": self.hash(),
        }


def build_plan(
    total_lengths: list[int],
    supervised_lengths: list[int],
    *,
    seed: int,
    epoch: int,
    max_tokens_per_microbatch: int,
    max_rows_per_microbatch: int,
    micro_batches_per_step: int,
    megabatch_rows: int,
    max_len: int,
    width_bucket: int = 1,
    drop_overlong: bool = True,
    order_mode: str = "reference_shuffle",
) -> Plan:
    if len(total_lengths) != len(supervised_lengths):
        raise ValueError("length arrays must match")
    config = {
        "seed": seed,
        "max_tokens_per_microbatch": max_tokens_per_microbatch,
        "max_rows_per_microbatch": max_rows_per_microbatch,
        "micro_batches_per_step": micro_batches_per_step,
        "megabatch_rows": megabatch_rows,
        "max_len": max_len,
        "width_bucket": width_bucket,
        "drop_overlong": drop_overlong,
        "n_rows_input": len(total_lengths),
    }
    if order_mode not in {"reference_shuffle", "curriculum"}:
        raise ValueError("order_mode must be reference_shuffle or curriculum")
    # Keep the historical reference plan hash byte-identical by recording this
    # field only for the new curriculum path.
    if order_mode != "reference_shuffle":
        config["order_mode"] = order_mode

    eligible = [i for i, n in enumerate(total_lengths)
                if n <= max_len or not drop_overlong]
    if max_len < 1:
        raise ValueError("max_len must be positive")
    if max_tokens_per_microbatch < max_len:
        raise ValueError(
            f"max_tokens_per_microbatch ({max_tokens_per_microbatch}) is below "
            f"max_len ({max_len}); no single long row could ever be placed")

    # Epoch-specific stream, derived so that epoch order never depends on how
    # many RNG draws the training loop happened to make.
    rng = random.Random((seed * 1_000_003) ^ (epoch + 1))
    order = list(eligible)
    if order_mode == "reference_shuffle":
        rng.shuffle(order)

    def bucketed(width: int) -> int:
        if width_bucket <= 1:
            return width
        return min(max_len, -(-width // width_bucket) * width_bucket)

    micro: list[MicroBatch] = []
    for start in range(0, len(order), megabatch_rows):
        mega = order[start:start + megabatch_rows]
        # Descending length, index as tiebreak -> total order, no ties to chance.
        mega.sort(key=lambda i: (-min(total_lengths[i], max_len), i))
        local_micro: list[MicroBatch] = []
        current: list[int] = []
        current_max = 0
        for idx in mega:
            row_len = min(total_lengths[idx], max_len)
            new_max = bucketed(max(current_max, row_len))
            fits = ((len(current) + 1) * new_max <= max_tokens_per_microbatch
                    and len(current) < max_rows_per_microbatch)
            if current and not fits:
                local_micro.append(_seal(current, current_max, supervised_lengths,
                                         max_len, total_lengths))
                current, current_max = [], 0
                new_max = bucketed(row_len)
            current.append(idx)
            current_max = new_max
        if current:
            local_micro.append(_seal(current, current_max, supervised_lengths,
                                     max_len, total_lengths))
        if order_mode == "curriculum":
            # Preserve coarse materialized curriculum buckets/megabatches while
            # avoiding a deterministic length sweep inside each one.
            rng.shuffle(local_micro)
        micro.extend(local_micro)

    if order_mode == "reference_shuffle":
        rng.shuffle(micro)

    steps = [Step(tuple(micro[i:i + micro_batches_per_step]))
             for i in range(0, len(micro), micro_batches_per_step)]
    # A step with zero supervised tokens cannot be normalised; it would divide by
    # zero rather than merely be useless.
    steps = [s for s in steps if s.supervised_tokens > 0]
    return Plan(steps=steps, epoch=epoch, config=config)


def _seal(indices: list[int], max_len_in_batch: int,
          supervised_lengths: list[int], max_len: int,
          total_lengths: list[int]) -> MicroBatch:
    supervised = sum(min(supervised_lengths[i], max_len) for i in indices)
    actual = sum(min(total_lengths[i], max_len) for i in indices)
    return MicroBatch(indices=tuple(indices), max_len=max_len_in_batch,
                      supervised=supervised, actual=actual)
