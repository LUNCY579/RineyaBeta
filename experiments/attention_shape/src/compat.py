#!/usr/bin/env python3
"""Workaround: `enable_gqa=True` silently falls back to the SDPA math kernel.

MEASURED ON THIS HOST (RTX 4070 Laptop, sm_89, torch 2.13.0+cu130,
transformers 5.15.0), q=16 heads / kv=8 heads / head_dim=128 / seq=16384, bf16:

    F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        -> tries to allocate 16.00 GiB  -> OOM
    repeat_kv(k, 2); F.scaled_dot_product_attention(q, k, v, is_causal=True)
        -> peak 224 MiB

16 GiB is exactly `16 heads x 16384 x 16384 x 4 bytes`: the full attention score
matrix in fp32, i.e. the math kernel. So `enable_gqa=True` does not reach the
flash or mem-efficient kernels here, and the cost is a factor of ~76 in memory.

`transformers.integrations.sdpa_attention.use_gqa_in_sdpa` opts into it whenever
`attention_mask is None`, with the comment that a mask "will fall back to the
math kernel" -- correct about masks, but the no-mask path is not safe either on
this stack. The condition is a property of the torch build and GPU, not of the
model, so it is not something the checkpoint can declare.

This matters far beyond this experiment: **any long-context work on this host
hits it**, and the failure mode is an OOM at a plausible-looking length rather
than an error naming the cause, so it reads like "the card is too small" when it
is really "the wrong kernel was chosen". The 8KV feasibility ceiling in
`docs/EXPERIMENT_DESIGN_KV_LONGCTX.md` was computed from cache bytes alone and
is *optimistic* until this patch is applied.

Kept as a local, opt-in monkey patch rather than a fork of transformers or an
edit to shared repo code, so nothing outside this experiment changes behaviour.

Usage:
    from compat import force_repeat_kv
    force_repeat_kv()          # call before building any model
"""

from __future__ import annotations

_PATCHED = False


def force_repeat_kv(verbose: bool = True) -> bool:
    """Make SDPA expand KV heads explicitly instead of using `enable_gqa`.

    Returns True if the patch was applied, False if it was already in place or
    the target could not be found (e.g. a transformers version that restructured
    the module -- in which case the caller should re-verify the memory claim
    rather than assume it is fixed).
    """
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from transformers.integrations import sdpa_attention as sa
    except ImportError:
        if verbose:
            print("compat: transformers sdpa_attention not importable; no patch applied")
        return False
    if not hasattr(sa, "use_gqa_in_sdpa"):
        if verbose:
            print("compat: use_gqa_in_sdpa missing -- transformers layout changed. "
                  "Re-verify the enable_gqa memory behaviour before trusting results.")
        return False

    sa.use_gqa_in_sdpa = lambda attention_mask, key, value: False
    _PATCHED = True
    if verbose:
        print("compat: enable_gqa disabled; SDPA will repeat_kv "
              "(avoids the fp32 math-kernel fallback)")
    return True


def verify_patch_matters(seq: int = 16384, n_q: int = 16, n_kv: int = 8,
                         head_dim: int = 128) -> dict:
    """Reproduce the measurement above so the claim can be re-checked, not trusted."""
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return {"skipped": "no cuda"}

    out: dict[str, object] = {"seq": seq, "n_q": n_q, "n_kv": n_kv}
    q = torch.randn(1, n_q, seq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, n_kv, seq, head_dim, device="cuda", dtype=torch.bfloat16)

    for label, fn in (
        ("enable_gqa", lambda: F.scaled_dot_product_attention(
            q, k, k, is_causal=True, enable_gqa=True)),
        ("repeat_kv", lambda: F.scaled_dot_product_attention(
            q, k.repeat_interleave(n_q // n_kv, dim=1),
            k.repeat_interleave(n_q // n_kv, dim=1), is_causal=True)),
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                fn()
            out[label] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        except torch.cuda.OutOfMemoryError:
            out[label] = "OOM"
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(verify_patch_matters(), indent=2))
