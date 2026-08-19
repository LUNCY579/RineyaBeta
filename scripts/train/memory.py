#!/usr/bin/env python3
"""Hard VRAM ceiling and a pre-flight probe that makes OOM a config error.

Strategy: refuse to start a run that could OOM, rather than recover from one.

Recovering from OOM means shrinking a batch, which changes the computation and
destroys bit-exact reproducibility. So instead:

  1. `apply_cap` gives the caching allocator a hard ceiling via
     `set_per_process_memory_fraction`. The process cannot grow past it even if
     the card is idle, so behaviour does not depend on what else is running.
  2. `probe` takes the *widest micro-batch the plan will ever produce* and runs a
     real forward and backward at exactly that shape, under the cap, before step
     zero. Peak allocation is measured, not estimated.
  3. If the peak exceeds the safety margin, the run aborts in seconds with a
     concrete `max_tokens_per_microbatch` to use instead -- not at hour six.

Because the plan's widest micro-batch bounds every other one, a passing probe
means no later step can allocate more for activations. The residual risks are
external: another process taking VRAM mid-run, or driver-level fragmentation.
Those surface as a clean resumable abort (see checkpoint.py), never as silent
recovery.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

MIB = 1 << 20


@dataclass
class ProbeReport:
    ran: bool
    device: str
    rows: int
    max_len: int
    padded_cost: int
    peak_alloc_mib: float
    peak_reserved_mib: float
    cap_mib: float
    margin_mib: float
    ok: bool
    suggestion: int | None = None

    def render(self) -> str:
        if not self.ran:
            return (f"memory probe skipped: device={self.device} "
                    f"(no CUDA; nothing to cap)")
        verdict = "OK" if self.ok else "OVER BUDGET"
        text = (
            f"memory probe [{verdict}]\n"
            f"  widest micro-batch : {self.rows} rows x {self.max_len} tokens "
            f"= {self.padded_cost:,d} padded positions\n"
            f"  peak allocated     : {self.peak_alloc_mib:,.0f} MiB\n"
            f"  peak reserved      : {self.peak_reserved_mib:,.0f} MiB\n"
            f"  cap                : {self.cap_mib:,.0f} MiB "
            f"(usable after margin: {self.cap_mib - self.margin_mib:,.0f} MiB)\n"
        )
        if not self.ok and self.rows > 1:
            text += (f"  -> lower max_tokens_per_microbatch to about "
                     f"{self.suggestion:,d} and re-run\n")
        elif not self.ok:
            # A single row cannot be split by shrinking the token budget, so the
            # usual advice would be actively misleading here.
            text += ("  -> the widest batch is ALREADY a single row of "
                     f"{self.max_len:,d} tokens, so lowering "
                     "max_tokens_per_microbatch cannot help.\n"
                     "     Remedies: raise memory.cap_mib; lower data.max_len "
                     "(truncates long rows); or accept nondeterministic attention "
                     "(mem-efficient SDPA) to drop the O(seq^2) score matrix.\n")
        return text.rstrip()


def device_total_mib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / MIB


def free_mib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(device)
    return free / MIB


def apply_cap(cap_mib: float | None, device: int = 0) -> float | None:
    """Install a hard allocator ceiling. Returns the cap actually applied."""
    if not torch.cuda.is_available() or cap_mib is None:
        return None
    total = device_total_mib(device)
    cap = min(float(cap_mib), total)
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    return cap


def probe(
    step_fn,
    *,
    rows: int,
    max_len: int,
    cap_mib: float | None,
    margin_mib: float = 512.0,
    device: int = 0,
) -> ProbeReport:
    """Run one synthetic forward+backward at the plan's worst-case shape.

    `step_fn(input_ids, attention_mask, labels)` must do a full forward and
    backward. Gradients are zeroed and caches dropped afterwards, and the caller
    is responsible for re-seeding: this deliberately perturbs nothing else, but
    it does consume RNG if the model has any stochastic layer.
    """
    if not torch.cuda.is_available():
        return ProbeReport(ran=False, device="cpu", rows=rows, max_len=max_len,
                           padded_cost=rows * max_len, peak_alloc_mib=0.0,
                           peak_reserved_mib=0.0, cap_mib=0.0,
                           margin_mib=margin_mib, ok=True)

    cap = cap_mib if cap_mib is not None else device_total_mib(device)
    usable = cap - margin_mib
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dev = torch.device(f"cuda:{device}")
    input_ids = torch.zeros((rows, max_len), dtype=torch.long, device=dev)
    attention_mask = torch.ones((rows, max_len), dtype=torch.long, device=dev)
    labels = input_ids.clone()

    oom = False
    try:
        step_fn(input_ids, attention_mask, labels)
    except torch.cuda.OutOfMemoryError:
        oom = True

    peak_alloc = torch.cuda.max_memory_allocated(device) / MIB
    peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
    del input_ids, attention_mask, labels
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    ok = (not oom) and peak_reserved <= usable
    suggestion = None
    if not ok:
        # Activation memory scales roughly with padded positions. Scale the budget
        # by the observed overshoot, with a conservative 10% extra haircut.
        ratio = usable / max(peak_reserved, 1.0)
        suggestion = max(max_len, int(rows * max_len * ratio * 0.9))
    return ProbeReport(ran=True, device=torch.cuda.get_device_name(device),
                       rows=rows, max_len=max_len, padded_cost=rows * max_len,
                       peak_alloc_mib=peak_alloc, peak_reserved_mib=peak_reserved,
                       cap_mib=cap, margin_mib=margin_mib, ok=ok,
                       suggestion=suggestion)


def snapshot(device: int = 0) -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "allocated_mib": round(torch.cuda.memory_allocated(device) / MIB, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / MIB, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / MIB, 1),
        "free_mib": round(free_mib(device), 1),
    }
