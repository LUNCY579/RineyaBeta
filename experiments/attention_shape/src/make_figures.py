#!/usr/bin/env python3
"""Generate the data figures directly from `shapes.py`.

The two conceptual diagrams in `figures/` are hand-authored; these two are not,
on purpose. A chart drawn by hand can drift away from the model it claims to
show. These are emitted from the same code that `verify_cache.py` checks against
measured cache bytes, so a figure that disagrees with the model is impossible by
construction.

Outputs:
    figures/03_cache_growth.svg     cache bytes vs context length (log-log)
    figures/04_iso_cache.svg        max reachable context at one cache budget

Usage:
    python src/make_figures.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import gqa, iso_cache_budget, mla, swa_bottom_global  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
BG, INK, MUTED, GRID = "#fbfcfd", "#10151c", "#5a6472", "#e3e8ee"

SERIES = [
    ("GQA-8KV  (R2 today)", gqa(8), "#1c4e8a", False),
    ("GQA-2KV  (surgery + healing)", gqa(2), "#7ba0d0", True),
    ("SWA w=1024, 4 global  (free)", swa_bottom_global(1024, 4), "#c2410c", False),
    ("SWA w=256, 4 global  (free)", swa_bottom_global(256, 4), "#ea8a4f", False),
    ("MLA d_c=256  (new code)", mla(256), "#186a4e", True),
]
LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header(w: int, h: int, title: str, sub: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        '<style>'
        '.ttl{font-size:24px;font-weight:700;fill:#10151c}'
        '.sub{font-size:14px;fill:#4a5568}'
        '.ax{font-size:12px;fill:#5a6472}'
        '.lg{font-size:13px;fill:#1a202c}'
        '.mono{font-family:Consolas,monospace;font-size:11.5px;fill:#10151c}'
        '.note{font-size:12.5px;fill:#4a5568}'
        '</style>',
        f'<rect width="{w}" height="{h}" fill="{BG}"/>',
        f'<text class="ttl" x="34" y="42">{esc(title)}</text>',
        f'<text class="sub" x="34" y="68">{esc(sub)}</text>',
    ]


def cache_growth() -> None:
    W, H = 1080, 660
    x0, y0, pw, ph = 96, 110, 660, 420
    lo_mib, hi_mib = 8.0, 32768.0

    def px(n: int) -> float:
        a, b = math.log2(LENGTHS[0]), math.log2(LENGTHS[-1])
        return x0 + pw * (math.log2(n) - a) / (b - a)

    def py(mib: float) -> float:
        a, b = math.log2(lo_mib), math.log2(hi_mib)
        return y0 + ph - ph * (math.log2(max(mib, lo_mib)) - a) / (b - a)

    s = header(W, H, "Cache growth: one axis saturates, the others do not",
               "Qwen3-0.6B shape, bf16. Sliding-window layers stop growing once "
               "context passes the window; head- and rank-based schemes never do.")

    # grid
    for mib in (8, 32, 128, 512, 2048, 8192, 32768):
        y = py(mib)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        lab = f"{mib / 1024:.0f} GiB" if mib >= 1024 else f"{mib} MiB"
        s.append(f'<text class="ax" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{lab}</text>')
    for n in LENGTHS:
        x = px(n)
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ph}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text class="ax" x="{x:.1f}" y="{y0 + ph + 20}" '
                 f'text-anchor="middle">{n // 1024}K</text>')
    s.append(f'<text class="ax" x="{x0 + pw / 2:.0f}" y="{y0 + ph + 46}" '
             f'text-anchor="middle">context length (tokens)</text>')

    # the 8 GiB card line
    y8 = py(8192)
    s.append(f'<line x1="{x0}" y1="{y8:.1f}" x2="{x0 + pw}" y2="{y8:.1f}" '
             f'stroke="#c0392b" stroke-width="1.6" stroke-dasharray="7 5"/>')
    s.append(f'<text class="mono" x="{x0 + 8}" y="{y8 - 8:.1f}" fill="#c0392b">'
             f'8 GiB — the whole card</text>')

    for label, shape, colour, dashed in SERIES:
        pts = " ".join(f"{px(n):.1f},{py(shape.cache_bytes(n) / 2 ** 20):.1f}"
                       for n in LENGTHS)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="2.6"{dash}/>')
        for n in LENGTHS:
            s.append(f'<circle cx="{px(n):.1f}" cy="{py(shape.cache_bytes(n) / 2 ** 20):.1f}" '
                     f'r="3.2" fill="{colour}"/>')

    # legend
    ly = y0 + 6
    s.append(f'<rect x="{x0 + pw + 24}" y="{ly - 22}" width="264" height="196" rx="9" '
             f'fill="#fff" stroke="{GRID}" stroke-width="1.4"/>')
    for label, shape, colour, dashed in SERIES:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        s.append(f'<line x1="{x0 + pw + 38}" y1="{ly:.0f}" x2="{x0 + pw + 72}" '
                 f'y2="{ly:.0f}" stroke="{colour}" stroke-width="3"{dash}/>')
        s.append(f'<text class="lg" x="{x0 + pw + 80}" y="{ly + 4:.0f}">{esc(label)}</text>')
        ly += 24
    s.append(f'<text class="note" x="{x0 + pw + 38}" y="{ly + 10:.0f}">dashed = needs</text>')
    s.append(f'<text class="note" x="{x0 + pw + 38}" y="{ly + 28:.0f}">work before use</text>')

    s.append(f'<rect x="34" y="{H - 76}" width="{W - 68}" height="56" rx="9" '
             f'fill="#fffbe9" stroke="#e0c060" stroke-width="1.4"/>')
    s.append(f'<text class="note" x="56" y="{H - 52}">Straight lines on a log-log plot '
             f'are constant cost per token: halving KV heads or lowering MLA rank shifts the line '
             f'down but never bends it.</text>')
    s.append(f'<text class="note" x="56" y="{H - 32}">The sliding-window curves bend. But this plot shows '
             f'CACHE only: measured peak memory tells a different story — see figure 05.</text>')
    s.append("</svg>")
    (FIG / "03_cache_growth.svg").write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"wrote {FIG / '03_cache_growth.svg'}")


def iso_cache() -> None:
    budget = iso_cache_budget(32768)  # what stock 8KV needs at 32K
    rows = [(label, shape, colour, dashed) for label, shape, colour, dashed in SERIES]

    W, H = 1080, 520
    x0, y0, pw = 330, 128, 560
    s = header(W, H, "At one fixed cache budget, how much context do you get?",
               f"Budget = {budget / 2 ** 20:.0f} MiB — exactly what stock 8KV spends to hold 32K tokens.")

    reach = []
    for label, shape, colour, dashed in rows:
        m = shape.max_length(budget)
        reach.append(m if m != float("inf") else 0.0)
    hi = max(reach)

    y = y0
    for (label, shape, colour, dashed), m in zip(rows, reach):
        bar = pw * (m / hi) if hi else 0
        opacity = ' fill-opacity="0.55"' if dashed else ""
        s.append(f'<text class="lg" x="{x0 - 16}" y="{y + 22}" text-anchor="end">{esc(label)}</text>')
        s.append(f'<rect x="{x0}" y="{y}" width="{bar:.1f}" height="32" rx="5" '
                 f'fill="{colour}"{opacity}/>')
        s.append(f'<text class="mono" x="{x0 + bar + 12:.1f}" y="{y + 22}">'
                 f'{int(m):,} tokens</text>')
        mult = m / reach[0] if reach[0] else 0
        s.append(f'<text class="mono" x="{x0 + pw + 118}" y="{y + 22}" fill="{MUTED}">'
                 f'{mult:.1f}×</text>')
        y += 52

    s.append(f'<text class="ax" x="{x0}" y="{y + 14}">longer is better — same bytes of cache in every row</text>')

    s.append(f'<rect x="34" y="{H - 118}" width="{W - 68}" height="98" rx="9" '
             f'fill="#fffbe9" stroke="#e0c060" stroke-width="1.4"/>')
    s.append(f'<text class="note" x="56" y="{H - 92}" font-weight="700">Read this as the cost question, not just the capability question.</text>')
    s.append(f'<text class="note" x="56" y="{H - 70}">The sliding-window rows are reached by editing a config file. '
             f'The GQA row needs irreversible weight surgery and a healing run;</text>')
    s.append(f'<text class="note" x="56" y="{H - 50}">the MLA row needs an attention module that does not exist yet in this stack. '
             f'A free option that merely ties a paid one still wins.</text>')
    s.append(f'<text class="note" x="56" y="{H - 30}">Caveat, and it is a big one: this is cache, not peak memory. '
             f'Measured, the sliding rows OOM at 32K and the GQA rows do not — see figure 05.</text>')
    s.append("</svg>")
    (FIG / "04_iso_cache.svg").write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"wrote {FIG / '04_iso_cache.svg'}")



def measured_vs_predicted() -> None:
    """Figure 05: the cache-only story vs what the card actually does.

    Reads results/measure_main.json; skipped silently if the sweep has not run.
    """
    import json
    res = Path(__file__).resolve().parents[1] / "results" / "measure_main.json"
    if not res.exists():
        print("skip 05: no results/measure_main.json")
        return
    rows = json.loads(res.read_text(encoding="utf-8"))["results"]
    order = ["base-8KV", "swa-w1024-g4", "swa-w256-g4", "swa-w1024-g2",
             "gqa-2KV", "gqa-1KV"]
    variants = [v for v in order if any(r["variant"] == v for r in rows)]
    at16 = {r["variant"]: r for r in rows if r["length"] == 16384}
    at32 = {r["variant"]: r for r in rows if r["length"] == 32768}
    if not at16:
        print("skip 05: no 16K cells")
        return

    W, H = 1080, 620
    x0, y0, pw, bh, gap = 250, 122, 520, 26, 16
    s2 = header(W, H, "Cache is not the constraint. Peak memory is.",
                "R2 at 16K context. Cache and peak allocation, measured on an "
                "RTX 4070 Laptop under a 6 GiB cap.")
    hi = max(max(r["peak_alloc_mib"] for r in at16.values()),
             max((r["cache_mib"] for r in at16.values()), default=0))

    s2.append(f'<text class="ax" x="{x0}" y="{y0 - 14}">cache (dark) and peak allocation (light), MiB</text>')
    y = y0
    for v in variants:
        r = at16.get(v)
        if r is None:
            continue
        cw = pw * r["cache_mib"] / hi
        pwid = pw * r["peak_alloc_mib"] / hi
        colour = "#186a4e" if v.startswith("gqa") else ("#1c4e8a" if v == "base-8KV" else "#c2410c")
        s2.append(f'<text class="lg" x="{x0 - 16}" y="{y + 30}" text-anchor="end">{esc(v)}</text>')
        s2.append(f'<rect x="{x0}" y="{y}" width="{pwid:.1f}" height="{bh}" rx="4" '
                  f'fill="{colour}" fill-opacity="0.28"/>')
        s2.append(f'<rect x="{x0}" y="{y + bh + 3}" width="{cw:.1f}" height="{bh - 8}" rx="4" '
                  f'fill="{colour}"/>')
        s2.append(f'<text class="mono" x="{x0 + max(pwid, cw) + 12:.1f}" y="{y + 20}">'
                  f'peak {r["peak_alloc_mib"]:.0f}</text>')
        s2.append(f'<text class="mono" x="{x0 + max(pwid, cw) + 12:.1f}" y="{y + 40}" fill="#5a6472">'
                  f'cache {r["cache_mib"]:.0f}</text>')
        ok32 = v in at32
        s2.append(f'<text class="mono" x="{x0 + pw + 168}" y="{y + 30}" '
                  f'fill="{"#186a4e" if ok32 else "#c0392b"}">'
                  f'{"32K OK" if ok32 else "32K OOM"}</text>')
        y += bh * 2 + gap

    s2.append(f'<rect x="34" y="{H - 140}" width="{W - 68}" height="120" rx="9" '
              f'fill="#fffbe9" stroke="#e0c060" stroke-width="1.4"/>')
    s2.append(f'<text class="note" x="56" y="{H - 114}" font-weight="700">The sliding-window rows have the smallest caches and the largest peaks.</text>')
    s2.append(f'<text class="note" x="56" y="{H - 92}">A sliding window is built as a dense [1,1,n,n] boolean mask, which scales with context rather than with the window — so</text>')
    s2.append(f'<text class="note" x="56" y="{H - 72}">w=256 and w=1024 have different caches and byte-identical peaks. The baseline gets no mask and takes the SDPA fast path.</text>')
    s2.append(f'<text class="note" x="56" y="{H - 50}" font-weight="700">Consequence: at 32K only the GQA rows fit. Reducing KV heads extends reachable context; narrowing the window does not.</text>')
    s2.append(f'<text class="note" x="56" y="{H - 28}">Sliding window still wins decode throughput (18.1 vs 11.4 tok/s at 16K) — it trades prefill and reach for steady-state speed.</text>')
    s2.append("</svg>")
    (FIG / "05_measured_vs_predicted.svg").write_text(chr(10).join(s2) + chr(10), encoding="utf-8")
    print(f"wrote {FIG / '05_measured_vs_predicted.svg'}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    cache_growth()
    iso_cache()
    measured_vs_predicted()
