#!/usr/bin/env python3
"""Crash-safe checkpointing with everything needed for bit-exact resume.

Two properties matter more than size here.

Atomicity. A checkpoint is written to a temporary path, flushed and fsynced, then
moved into place with `os.replace`, which is atomic on both POSIX and Windows. A
kill -9 or power loss mid-write therefore leaves the *previous* checkpoint intact
rather than a truncated file that loads without error and trains from garbage. The
`latest.json` pointer is updated the same way, and only after the payload landed.

Completeness. Bit-exact resume needs more than weights and optimizer state. It
needs every RNG stream, the exact position in the plan, and the identity of the
plan itself. If the plan hash on disk does not match the plan regenerated at
resume time, the run refuses to continue: the alternative is silently training a
different model than the one the checkpoint describes.

Guard rails deliberately enforced on load:
  * plan hash mismatch     -> refuse (config or batching changed under the run)
  * mix sha256 mismatch    -> refuse (training data changed)
  * code fingerprint drift -> refuse (the plan hash covers *what* the optimizer
                              sees, not the code consuming it; editing the loss
                              would otherwise satisfy every other guard while
                              silently changing the numerics)
  * environment drift      -> refuse (different GPU/driver/torch means a
                              different reduction order, so a different model)

Each can be overridden explicitly via the `resume` config block, never silently.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from determinism import (RngState, code_fingerprint, compare_code_fingerprint,
                         compare_fingerprint, fingerprint)


@dataclass
class Position:
    epoch: int
    step_in_epoch: int      # next step index to execute within the epoch plan
    global_step: int
    rows_seen: int
    supervised_tokens_seen: int


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save(
    out_dir: Path,
    *,
    position: Position,
    adapter_state: dict,
    optimizer_state: dict,
    scheduler_state: dict | None,
    rng: RngState,
    plan_hash: str,
    plan_summary: dict,
    mix_sha256: str,
    config: dict,
    metrics: dict,
    keep_last: int = 3,
    extra_state: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{position.global_step:09d}.pt"
    payload = {
        "position": asdict(position),
        "adapter_state": adapter_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng": {"python": rng.python, "numpy": rng.numpy,
                "torch_cpu": rng.torch_cpu, "torch_cuda": rng.torch_cuda},
        "plan_hash": plan_hash,
        "plan_summary": plan_summary,
        "mix_sha256": mix_sha256,
        "config": config,
        "metrics": metrics,
        "fingerprint": fingerprint(),
        "code_fingerprint": code_fingerprint(),
        # Legacy healing checkpoints remain format 1.  RGE uses the optional
        # extension for optimizer/geometry contract state and is format 2.
        "format_version": 2 if extra_state is not None else 1,
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _atomic_write_bytes(out_dir / name, buffer.getvalue())

    # Pointer is written only after the payload is durable.
    _atomic_write_text(out_dir / "latest.json", json.dumps({
        "checkpoint": name,
        "global_step": position.global_step,
        "epoch": position.epoch,
        "step_in_epoch": position.step_in_epoch,
        "plan_hash": plan_hash,
        "mix_sha256": mix_sha256,
        "metrics": metrics,
    }, indent=2))

    keep = sorted(out_dir.glob("step_*.pt"))
    for stale in keep[:-keep_last] if keep_last > 0 else []:
        stale.unlink(missing_ok=True)
    return out_dir / name


def latest_path(out_dir: Path) -> Path | None:
    pointer = out_dir / "latest.json"
    if not pointer.exists():
        return None
    try:
        name = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    except (json.JSONDecodeError, KeyError):
        return None
    candidate = out_dir / name
    if candidate.exists():
        return candidate
    # Pointer lost or truncated: fall back to the newest intact payload.
    payloads = sorted(out_dir.glob("step_*.pt"))
    return payloads[-1] if payloads else None


def load(path: Path, *, expect_plan_hash: str, expect_mix_sha256: str,
         allow_env_drift: bool = False, allow_code_drift: bool = False) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("plan_hash") != expect_plan_hash:
        raise SystemExit(
            f"plan hash mismatch in {path.name}.\n"
            f"  checkpoint: {payload.get('plan_hash')}\n"
            f"  regenerated: {expect_plan_hash}\n"
            "The batch plan changed, so resuming would not reproduce the run. "
            "Either restore the original config (seed, budgets, max_len, mix) or "
            "start a new run directory."
        )
    if payload.get("mix_sha256") != expect_mix_sha256:
        raise SystemExit(
            f"training mix changed.\n  checkpoint: {payload.get('mix_sha256')}\n"
            f"  on disk   : {expect_mix_sha256}\nRefusing to resume."
        )

    code_drift = compare_code_fingerprint(payload.get("code_fingerprint", {}),
                                          code_fingerprint())
    if code_drift:
        message = ("training source changed since this checkpoint was written; "
                   "resuming would not reproduce the run:\n  - "
                   + "\n  - ".join(code_drift))
        if not allow_code_drift:
            raise SystemExit(message + "\nRestore the original source, start a new "
                                       "run directory, or set "
                                       "resume.allow_code_drift=true if the change "
                                       "is known to be inert (comments only).")
        print("WARNING: " + message)

    drift = compare_fingerprint(payload.get("fingerprint", {}), fingerprint())
    if drift:
        message = ("environment drift detected; byte-identical replay is NOT "
                   "guaranteed:\n  - " + "\n  - ".join(drift))
        if not allow_env_drift:
            raise SystemExit(message + "\nPass resume.allow_env_drift=true to "
                                       "continue anyway (results will differ).")
        print("WARNING: " + message)

    payload["rng_state"] = RngState(
        python=payload["rng"]["python"],
        numpy=payload["rng"]["numpy"],
        torch_cpu=payload["rng"]["torch_cpu"],
        torch_cuda=payload["rng"]["torch_cuda"],
    )
    payload["position_obj"] = Position(**payload["position"])
    return payload
