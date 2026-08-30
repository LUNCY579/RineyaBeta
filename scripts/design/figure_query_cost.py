#!/usr/bin/env python3
"""Figure: why the query-aware pass costs 8-11x what its docstring predicted.

WHY A SCRIPT AND NOT AN IMAGE, restated from `figure_keep_fraction.py` because it
is the rule and not a preference: a figure is a number the reader cannot check by
grepping, so it must be generated from the same committed JSON that
`verify_claims.py` reads. Drawing it by hand would create the one artefact in the
paper whose provenance is "I typed it".

WHAT IT SHOWS. Measured forward cost against token count, with two models drawn
over it:

  - the PROPORTIONAL model the `score_with_query` docstring assumed, a line
    through the origin calibrated on the largest measured point. This is what
    "16 tokens against a 4096-token chunk is 0.4% of the work" means.
  - the FITTED model t(T) = a + b*T, which reproduces the measurements at
    R^2 > 0.996.

The two are indistinguishable at T = 512 and differ by ~5x at T = 16. That gap is
the entire finding: a short forward is bound by streaming the weights, not by the
arithmetic on its tokens, so the pass does not get cheap just because the question
is short.

No matplotlib: SVG is emitted directly, so this adds no dependency and the output
is vector for print.

Usage:
    python scripts/design/figure_query_cost.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KVS = ROOT / "experiments/kv_significance/results"

W, H = 760, 470
L, R, T, B = 92, 132, 52, 76
PW, PH = W - L - R, H - T - B

X_HI = 540.0
Y_HI = 1400.0

INK = "#1a1a1a"
GRID = "#e8e8e8"
MEAS = "#1f4e79"
FIT = "#1f4e79"
NAIVE = "#b03a2e"
FLOOR = "#f2c14e"


def x(t: float) -> float:
    return L + (t / X_HI) * PW


def y(ms: float) -> float:
    return T + PH - (ms / Y_HI) * PH


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(doc: dict) -> str:
    cl = doc["component_ladder"]
    pts = [(r["tokens"], r["ms_min"]) for r in cl["rows"]]
    a = cl["fit_intercept_ms"]
    b = cl["fit_slope_ms_per_token"]
    r2 = cl["fit_r2"]
    t_max, ms_max = pts[-1]
    # The docstring's model: cost proportional to token count. Calibrated on the
    # largest measured point, which is the most generous reading of it -- it is
    # allowed to be exactly right where forwards are long.
    slope_naive = ms_max / t_max

    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'viewBox="0 0 %d %d" font-family="Georgia,\'Times New Roman\',serif">'
         % (W, H, W, H),
         '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H)]

    # --- the fixed floor, drawn as a band so it reads as "cost you pay anyway"
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" '
             'fill-opacity="0.20"/>' % (L, y(a), PW, (T + PH) - y(a), FLOOR))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.4" stroke-dasharray="6,3"/>'
             % (L, y(a), L + PW, y(a), "#c9971f"))
    # Inside the band, not above it: the measured points at T = 8 and 16 sit
    # within 10 px of the band's top edge and a label there collides with them.
    o.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="#8a6a12" '
             'font-style="italic">fixed floor %.0f ms — every weight streamed '
             'once, whatever T is</text>' % (L + 8, y(a) + 17, a))

    # --- axes and gridlines
    for gv in range(0, int(Y_HI) + 1, 200):
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                 'stroke-width="1"/>' % (L, y(gv), L + PW, y(gv), GRID))
        o.append('<text x="%.1f" y="%.1f" text-anchor="end" font-size="12" '
                 'fill="#555">%d</text>' % (L - 9, y(gv) + 4, gv))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.3"/>' % (L, T, L, T + PH, "#999"))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.3"/>' % (L, T + PH, L + PW, T + PH, "#999"))
    for tv in (0, 128, 256, 384, 512):
        o.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="12" '
                 'fill="#555">%d</text>' % (x(tv), T + PH + 20, tv))

    # --- the two models
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="2" stroke-dasharray="8,4"/>'
             % (x(0), y(0), x(X_HI), y(slope_naive * X_HI), NAIVE))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="2"/>'
             % (x(0), y(a), x(X_HI), y(a + b * X_HI), FIT))

    # --- the measurements
    for tv, ms in pts:
        o.append('<circle cx="%.1f" cy="%.1f" r="4.6" fill="%s"/>'
                 % (x(tv), y(ms), MEAS))

    # --- the gap at Q = 16, which is the finding
    q = 16
    got, pred = a + b * q, slope_naive * q
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.6"/>' % (x(q), y(pred), x(q), y(got), "#444"))
    o.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="%s"/>'
             % (x(q), y(pred), NAIVE))
    # The annotation goes in the empty upper-left quadrant, not beside the
    # marker: both fitted and measured curves pass through the region just right
    # of T = 16, so text anchored there crosses them.
    o.append('<text x="%.1f" y="%.1f" font-size="13" fill="%s">'
             'At a 16-token question the proportional model predicts '
             '<tspan fill="%s">%.0f ms</tspan>.</text>'
             % (L + 18, 92, INK, NAIVE, pred))
    o.append('<text x="%.1f" y="%.1f" font-size="13" fill="%s">'
             'The measurement is <tspan font-weight="bold">%.0f ms</tspan> — '
             '%.1f× more, and %.0f%% of it is the floor.</text>'
             % (L + 18, 111, INK, got, got / pred, 100 * a / got))
    o.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="#444" '
             'font-weight="bold">%.1f×</text>'
             % (x(q) + 9, (y(pred) + y(got)) / 2 + 4, got / pred))

    # --- legend
    lx, ly = L + PW + 14, T + 116
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="2"/>' % (lx, ly, lx + 22, ly, FIT))
    o.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="#333">fitted</text>'
             % (lx, ly + 16))
    o.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="#666">'
             'a + b·T</text>' % (lx, ly + 30))
    o.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="#666">'
             'R² = %.4f</text>' % (lx, ly + 44, r2))
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="2" stroke-dasharray="8,4"/>'
             % (lx, ly + 70, lx + 22, ly + 70, NAIVE))
    o.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="#333">'
             'assumed</text>' % (lx, ly + 86))
    o.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="#666">'
             'cost ∝ tokens</text>' % (lx, ly + 100))

    # --- titles
    o.append('<text x="%.1f" y="%.1f" font-size="16" fill="%s">'
             'A short forward is not a small forward</text>' % (L, 26, INK))
    o.append('<text x="%.1f" y="%.1f" font-size="12" fill="#666">'
             'forward cost against token count, %s cache, %s %s, '
             'minimum of %d interleaved rounds</text>'
             % (L, 42, f'{cl["cache_len"]:,}-position', doc["dtype"],
                doc["device"].upper(), doc.get("rounds_components", doc["rounds"])))
    o.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="12.5" '
             'fill="#333">tokens in the forward (T)</text>'
             % (L + PW / 2, T + PH + 44))
    o.append('<text transform="translate(%.1f,%.1f) rotate(-90)" '
             'text-anchor="middle" font-size="12.5" fill="#333">'
             'wall-clock (ms)</text>' % (24, T + PH / 2))
    o.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="#888">'
             'generated by scripts/design/figure_query_cost.py from '
             'experiments/kv_significance/results/measure_query_prefill.json'
             '</text>' % (L, H - 12))
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=KVS / "measure_query_prefill.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/figures/query_pass_cost.svg")
    args = ap.parse_args()

    doc = json.loads(args.src.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(doc), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
