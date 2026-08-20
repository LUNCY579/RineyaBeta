#!/usr/bin/env python3
"""Analytic KV-cache cost for every attention shape under study.

One module, one contract: given an architecture and a context length, return the
exact number of cache bytes. Everything else in this experiment -- the iso-cache
comparisons, the feasibility checks, the Pareto plots -- is built on this, so it
is kept pure (no torch, no I/O) and is cross-checked against *measured* cache
bytes by `verify_cache.py`. An analytic model nobody checked is a guess.

Four shapes, three of them reachable on this host today:

  GQA   grouped-query attention        cache = 2 * L * g * d_head  per token
  SWA   sliding-window hybrid          cache saturates at the window; per-layer
  CLA   cross-layer KV sharing         GQA divided by the sharing factor
  MLA   multi-head latent attention    cache = L * (d_c + d_rope) per token

The important structural difference is that GQA/CLA/MLA all cost a fixed number
of bytes *per token*, so cache grows linearly forever, while SWA saturates: a
sliding layer stops growing once the context passes its window. A model whose
layers are all sliding has an O(1) cache and therefore unbounded context at
constant memory -- at the cost of having no path for information to travel
further than the window. That trade is the subject of this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Qwen3-0.6B / R2 shape. Both share it; only vocab differs, which does not
# touch the cache.
QWEN3_06B = dict(n_layers=28, n_q_heads=16, kv_heads=8, head_dim=128, hidden=1024)


@dataclass(frozen=True)
class Shape:
    """An attention configuration and its cache cost."""

    name: str
    n_layers: int = 28
    kv_heads: int = 8
    head_dim: int = 128
    bytes_per_elem: int = 2
    # Per-layer window; None means full attention on that layer.
    windows: tuple[int | None, ...] = field(default=())
    # Cross-layer sharing factor: 1 = no sharing, 2 = each KV set serves 2 layers.
    share: int = 1
    # MLA latent dims; when set, supersedes the GQA cache term.
    d_latent: int | None = None
    d_rope: int = 64

    def __post_init__(self) -> None:
        if self.windows and len(self.windows) != self.n_layers:
            raise ValueError(
                f"{self.name}: {len(self.windows)} windows for {self.n_layers} layers"
            )
        if self.n_layers % self.share:
            raise ValueError(f"{self.name}: share={self.share} does not divide {self.n_layers}")

    # -- cost ---------------------------------------------------------------

    def per_layer_dims(self) -> int:
        """Cached dimensions for one layer, one token."""
        if self.d_latent is not None:
            return self.d_latent + self.d_rope
        return 2 * self.kv_heads * self.head_dim

    def cache_bytes(self, length: int) -> int:
        """Total cache bytes holding `length` tokens."""
        per_layer = self.per_layer_dims() * self.bytes_per_elem
        if not self.windows:
            tokens_held = self.n_layers * length
        else:
            tokens_held = sum(length if w is None else min(length, w) for w in self.windows)
        return per_layer * tokens_held // self.share

    def bytes_per_token(self, length: int) -> float:
        """Marginal-ish cost; for SWA this falls as the window saturates."""
        return self.cache_bytes(length) / length if length else 0.0

    def saturates(self) -> bool:
        """True when cache is bounded regardless of context length."""
        return bool(self.windows) and all(w is not None for w in self.windows)

    def max_length(self, budget_bytes: int) -> float:
        """Longest context fitting in `budget_bytes`. inf when the cache saturates."""
        if self.saturates():
            return float("inf") if self.cache_bytes(10**9) <= budget_bytes else 0.0
        # Bisect: cache_bytes is monotone non-decreasing in length.
        lo, hi = 0, 1
        while self.cache_bytes(hi) <= budget_bytes:
            hi *= 2
            if hi > 2 ** 34:
                return float(hi)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.cache_bytes(mid) <= budget_bytes:
                lo = mid
            else:
                hi = mid
        return float(lo)

    def n_global_layers(self) -> int:
        return self.n_layers if not self.windows else sum(w is None for w in self.windows)


# -- constructors -----------------------------------------------------------


def gqa(kv_heads: int, **kw) -> Shape:
    return Shape(name=f"GQA-{kv_heads}KV", kv_heads=kv_heads, **kw)


def swa_bottom_global(window: int, n_global: int, kv_heads: int = 8, n_layers: int = 28,
                      **kw) -> Shape:
    """`max_window_layers` semantics: layers with index >= n_global slide.

    This mirrors `Qwen3Config` exactly -- the bottom `n_global` layers keep full
    attention and everything above slides -- so a Shape built this way and a
    config built by `make_variants.py` describe the same model.
    """
    windows = tuple(None if i < n_global else window for i in range(n_layers))
    return Shape(name=f"SWA-w{window}-g{n_global}", kv_heads=kv_heads,
                 n_layers=n_layers, windows=windows, **kw)


def swa_interleaved(window: int, period: int, kv_heads: int = 8, n_layers: int = 28,
                    **kw) -> Shape:
    """Every `period`-th layer is global (Gemma-style), the rest slide.

    Requires an explicit `layer_types` list in the config; `max_window_layers`
    cannot express it.
    """
    windows = tuple(None if (i + 1) % period == 0 else window for i in range(n_layers))
    n_glob = sum(w is None for w in windows)
    return Shape(name=f"SWA-w{window}-every{period}(g{n_glob})", kv_heads=kv_heads,
                 n_layers=n_layers, windows=windows, **kw)


def cla(share: int, kv_heads: int = 8, **kw) -> Shape:
    return Shape(name=f"CLA-{share}x-{kv_heads}KV", kv_heads=kv_heads, share=share, **kw)


def mla(d_latent: int, d_rope: int = 64, **kw) -> Shape:
    return Shape(name=f"MLA-d{d_latent}+{d_rope}", d_latent=d_latent, d_rope=d_rope, **kw)


def iso_cache_budget(length: int, reference: Shape | None = None) -> int:
    """Bytes the stock 8KV model would spend at `length` -- the iso-cache yardstick."""
    return (reference or gqa(8)).cache_bytes(length)


LADDER = (4096, 16384, 32768, 65536, 131072)


def main() -> None:
    catalogue: list[Shape] = [
        gqa(8), gqa(4), gqa(2), gqa(1),
        swa_bottom_global(1024, 4), swa_bottom_global(1024, 7),
        swa_bottom_global(4096, 4), swa_interleaved(1024, 4),
        cla(2), cla(4), cla(2, kv_heads=4),
        mla(512), mla(256), mla(128),
    ]

    print("cache bytes at each context length (MiB), Qwen3-0.6B shape, bf16")
    head = f"{'shape':26} {'glob':>4} " + "".join(f"{n // 1024:>8}K" for n in LADDER)
    print(head)
    for s in catalogue:
        cells = "".join(f"{s.cache_bytes(n) / 2**20:>9.0f}" for n in LADDER)
        print(f"{s.name:26} {s.n_global_layers():>4} {cells}")

    print("\nreduction vs stock 8KV, at each length")
    print(head)
    ref = gqa(8)
    for s in catalogue:
        cells = "".join(f"{ref.cache_bytes(n) / s.cache_bytes(n):>8.1f}x" for n in LADDER)
        print(f"{s.name:26} {s.n_global_layers():>4} {cells}")

    # Iso-cache: everyone gets the bytes stock 8KV needs at 32K.
    budget = iso_cache_budget(32768)
    print(f"\niso-cache frontier at B = {budget / 2**20:.0f} MiB "
          f"(what stock 8KV needs for 32K)")
    print(f"{'shape':26} {'max context':>14}")
    for s in catalogue:
        m = s.max_length(budget)
        txt = "unbounded" if m == float("inf") else f"{int(m):,}"
        print(f"{s.name:26} {txt:>14}")


if __name__ == "__main__":
    main()
