#!/usr/bin/env python3
"""Generate the paper's main figure from committed results.

WHY A SCRIPT AND NOT AN IMAGE. Every number in this work is traceable to a
committed result file, and a figure is a number the reader cannot check by
grepping. Drawing it by hand would create the one artefact in the paper whose
provenance is "I typed it". This reads the same JSON that verify_claims reads and
emits SVG, so the figure cannot drift from the data behind it.

WHAT IT SHOWS. Accuracy against the KEEP FRACTION, with context length as the
series. The paper's central claim is that the length is not the variable -- the
two ladders lie on top of each other and saturate at the same keep fraction -- so
the figure is drawn on the axis that makes that visible, rather than on length.

The 32K point is plotted even though it is awkward: at 6.25% keep it sits BELOW
both longer contexts. Omitting an inconvenient point would be the one thing this
program has spent its whole length trying not to do. It is drawn as an isolated
marker because 32K has a single compressed cell, not a ladder.

No matplotlib: SVG is emitted directly, so this adds no dependency to the
project and the output is vector for print.

Usage:
    python scripts/design/figure_keep_fraction.py
    python scripts/design/figure_keep_fraction.py --out docs/figures/fig1.svg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KVS = ROOT / "experiments/kv_significance/results"

W, H = 760, 460
L, R, T, B = 88, 34, 46, 74          # margins
PW, PH = W - L - R, H - T - B

X_LO, X_HI = 1.2, 20.0               # keep fraction, per cent, log scale
Y_LO, Y_HI = 0.0, 1.06


def x(pct: float) -> float:
    f = (math.log10(pct) - math.log10(X_LO)) / (math.log10(X_HI) - math.log10(X_LO))
    return L + f * PW


def y(acc: float) -> float:
    return T + PH - (acc - Y_LO) / (Y_HI - Y_LO) * PH


def cells(fname: str, length: int, budgets=None):
    """(keep_pct, accuracy) for each budget in a committed result file."""
    p = KVS / fname
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for r in d["results"]:
        if r["length"] != length or r.get("acc") is None or not r["budget"]:
            continue
        if budgets and r["budget"] not in budgets:
            continue
        out.setdefault(r["budget"], []).append(r["acc"])
    return sorted((100.0 * b / length, sum(v) / len(v)) for b, v in out.items())


def merge(*series):
    seen: dict[float, float] = {}
    for s in series:
        for k, v in s:
            seen[round(k, 4)] = v
    return sorted(seen.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/keep_fraction.svg")
    args = ap.parse_args()

    s64 = merge(cells("pk_64k_ladder.json", 65536),
                cells("pk_ladder_65536_B8192.json", 65536))
    s128 = merge(cells("pk_128k_grid12.json", 131072),
                 cells("pk_128k_B4096.json", 131072),
                 cells("pk_ladder_131072_B16384.json", 131072))
    s32 = merge(cells("pk_32k_paired.json", 32768))
    if not s64 or not s128:
        raise SystemExit("missing ladder data; run the passkey ladders first")

    o = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Georgia,\'Times New Roman\',serif">')
    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # --- the saturated reference, which is a bound rather than a measurement ---
    add(f'<line x1="{L}" y1="{y(1.0):.1f}" x2="{L+PW}" y2="{y(1.0):.1f}" '
        f'stroke="#b0b0b0" stroke-width="1.2" stroke-dasharray="7,4"/>')
    add(f'<text x="{L+PW-4:.0f}" y="{y(1.0)-8:.1f}" text-anchor="end" '
        f'font-size="12.5" fill="#666" font-style="italic">'
        f'uncompressed reference, bounded at 1.00 by saturation</text>')

    # --- axes -----------------------------------------------------------------
    for acc in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yy = y(acc)
        add(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+PW}" y2="{yy:.1f}" '
            f'stroke="#ededed" stroke-width="1"/>')
        add(f'<text x="{L-11}" y="{yy+4:.1f}" text-anchor="end" font-size="13" '
            f'fill="#333">{acc:.1f}</text>')
    for pct, lab in ((1.56, "1.56"), (3.12, "3.12"), (6.25, "6.25"), (12.5, "12.5")):
        xx = x(pct)
        add(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T+PH}" '
            f'stroke="#f4f4f4" stroke-width="1"/>')
        add(f'<text x="{xx:.1f}" y="{T+PH+22}" text-anchor="middle" font-size="13" '
            f'fill="#333">{lab}%</text>')
    add(f'<line x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}" stroke="#333" stroke-width="1.4"/>')
    add(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}" stroke="#333" stroke-width="1.4"/>')
    add(f'<text x="{L+PW/2:.0f}" y="{H-26}" text-anchor="middle" font-size="14.5" '
        f'fill="#111">KV cache retained (per cent of context)</text>')
    add(f'<text transform="translate(24,{T+PH/2:.0f}) rotate(-90)" text-anchor="middle" '
        f'font-size="14.5" fill="#111">passkey retrieval accuracy</text>')

    def series(pts, colour, dash, marker, label):
        d = " ".join(("M" if i == 0 else "L") + f"{x(k):.1f},{y(v):.1f}"
                     for i, (k, v) in enumerate(pts))
        add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.4" '
            f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for k, v in pts:
            if marker == "circle":
                add(f'<circle cx="{x(k):.1f}" cy="{y(v):.1f}" r="5" fill="#fff" '
                    f'stroke="{colour}" stroke-width="2.4"/>')
            else:
                s = 4.6
                add(f'<rect x="{x(k)-s:.1f}" y="{y(v)-s:.1f}" width="{2*s}" '
                    f'height="{2*s}" fill="#fff" stroke="{colour}" stroke-width="2.4"/>')

    series(s64, "#1a5fb4", "none", "circle", "64K")
    series(s128, "#a51d2d", "none", "square", "128K")

    for k, v in s32:                       # single cell: marker only, no line
        add(f'<path d="M{x(k):.1f},{y(v)-6:.1f} L{x(k)+6:.1f},{y(v):.1f} '
            f'L{x(k):.1f},{y(v)+6:.1f} L{x(k)-6:.1f},{y(v):.1f} Z" '
            f'fill="#fff" stroke="#6a6a6a" stroke-width="2.2"/>')

    # --- the point of the figure ---------------------------------------------
    xs, ys = x(12.5), y(1.0)
    add(f'<circle cx="{xs:.1f}" cy="{ys:.1f}" r="13" fill="none" stroke="#2a7d2a" '
        f'stroke-width="1.8"/>')
    add(f'<text x="{xs-20:.1f}" y="{ys-20:.1f}" text-anchor="end" font-size="13.5" '
        f'fill="#2a7d2a" font-weight="bold">lossless at both lengths</text>')

    # --- legend ---------------------------------------------------------------
    lx, ly = L + 18, T + 16
    for i, (col, mk, lab) in enumerate((("#1a5fb4", "circle", "64K context"),
                                        ("#a51d2d", "square", "128K context"),
                                        ("#6a6a6a", "diamond", "32K context (one cell)"))):
        yy = ly + i * 21
        if mk == "circle":
            add(f'<circle cx="{lx}" cy="{yy-4}" r="5" fill="#fff" stroke="{col}" stroke-width="2.4"/>')
        elif mk == "square":
            add(f'<rect x="{lx-4.6}" y="{yy-8.6}" width="9.2" height="9.2" fill="#fff" '
                f'stroke="{col}" stroke-width="2.4"/>')
        else:
            add(f'<path d="M{lx},{yy-10} L{lx+6},{yy-4} L{lx},{yy+2} L{lx-6},{yy-4} Z" '
                f'fill="#fff" stroke="{col}" stroke-width="2.2"/>')
        add(f'<text x="{lx+14}" y="{yy}" font-size="13" fill="#222">{lab}</text>')
    add('</svg>')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")
    for name, s in (("64K", s64), ("128K", s128), ("32K", s32)):
        print(f"  {name:>5}: " + "  ".join(f"{k:.2f}%->{v:.2f}" for k, v in s))


if __name__ == "__main__":
    main()
