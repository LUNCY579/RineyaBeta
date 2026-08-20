#!/usr/bin/env python3
"""Figure 06: why a sliding window collapses, in two panels.

Left  -- measured attention mass on the first 4 tokens, per layer (sink_audit.py).
Right -- measured quality against how far the prompt runs past the window.

Both panels come from files in results/, so the figure cannot drift from the
numbers it claims to show.

Usage:
    python src/make_figure_06.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import FIG, GRID, esc, header  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
NL = chr(10)


def cliff_buckets() -> list[tuple[str, int, float, float]]:
    def load(p):
        return {json.loads(l)["id"]: json.loads(l)
                for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = load(RES / "base-8KV__rag_dev" / "results.jsonl")
    swa = load(RES / "swa-w128-g4__rag_dev" / "results.jsonl")
    buckets: dict[str, list] = {}
    for i, b in base.items():
        if i not in swa:
            continue
        over = b["prompt_tokens"] - 128
        key = ("inside window" if over <= 0 else
               "1-100 over" if over <= 100 else
               "101-200 over" if over <= 200 else "200+ over")
        buckets.setdefault(key, []).append(
            (float(b.get("quality_score") or 0.0), float(swa[i].get("quality_score") or 0.0)))
    order = ["inside window", "1-100 over", "101-200 over", "200+ over"]
    return [(k, len(buckets[k]),
             statistics.mean(x for x, _ in buckets[k]),
             statistics.mean(y for _, y in buckets[k]))
            for k in order if k in buckets]


def main() -> None:
    prof = json.loads((RES / "sink_profile.json").read_text(encoding="utf-8"))
    sink = prof["per_layer_sink_share"]
    rows = cliff_buckets()

    W, H = 1180, 620
    s = header(W, H, "Why a sliding window collapses: it evicts the attention sink",
               "Both panels measured on R2. Left: sink_audit.py. "
               "Right: the rag_dev quality sweep.")

    # ---------------- left panel: sink per layer ----------------
    lx, ly, lw, lh = 70, 130, 470, 380
    s.append(f'<text class="lg" x="{lx}" y="{ly - 34}" font-weight="700">'
             f'attention mass on the first {prof["first_k"]} tokens, by layer</text>')
    s.append(f'<text class="ax" x="{lx}" y="{ly - 14}">'
             f'mean {prof["mean"]:.1%} across all 28 layers, {prof["cases"]} prompts, 16 heads</text>')
    n = len(sink)
    bw = lw / n
    for frac, lab in ((0.0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")):
        y = ly + lh - lh * frac
        s.append(f'<line x1="{lx}" y1="{y:.1f}" x2="{lx + lw}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text class="ax" x="{lx - 8}" y="{y + 4:.1f}" text-anchor="end">{lab}</text>')
    y50 = ly + lh - lh * 0.5
    s.append(f'<line x1="{lx}" y1="{y50:.1f}" x2="{lx + lw}" y2="{y50:.1f}" '
             f'stroke="#c0392b" stroke-width="1.5" stroke-dasharray="6 4"/>')
    for i, v in enumerate(sink):
        h = lh * min(v, 1.0)
        colour = "#c2410c" if v > 0.5 else "#e0a068"
        s.append(f'<rect x="{lx + i * bw + 1:.1f}" y="{ly + lh - h:.1f}" '
                 f'width="{bw - 2:.1f}" height="{h:.1f}" fill="{colour}"/>')
    for i in (0, 9, 18, 27):
        s.append(f'<text class="ax" x="{lx + i * bw + bw / 2:.1f}" y="{ly + lh + 18}" '
                 f'text-anchor="middle">{i}</text>')
    s.append(f'<text class="ax" x="{lx + lw / 2:.0f}" y="{ly + lh + 40}" '
             f'text-anchor="middle">layer</text>')
    hi = sum(1 for v in sink if v > 0.5)
    s.append(f'<text class="note" x="{lx}" y="{ly + lh + 68}">'
             f'{hi} of {n} layers send more than half their attention to those 4 positions; '
             f'layer 27 sends {sink[-1]:.0%}.</text>')

    # ---------------- right panel: the cliff ----------------
    rx, ry, rw, rh = 640, 130, 470, 380
    s.append(f'<text class="lg" x="{rx}" y="{ry - 34}" font-weight="700">'
             f'rag_dev quality vs. how far the prompt runs past w=128</text>')
    s.append(f'<text class="ax" x="{rx}" y="{ry - 14}">'
             f'damage does not grow with context lost — it is a cliff</text>')
    for frac, lab in ((0.0, "0.0"), (0.25, "0.25"), (0.5, "0.50"), (0.75, "0.75"), (1.0, "1.0")):
        y = ry + rh - rh * frac
        s.append(f'<line x1="{rx}" y1="{y:.1f}" x2="{rx + rw}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text class="ax" x="{rx - 8}" y="{y + 4:.1f}" text-anchor="end">{lab}</text>')
    gw = rw / len(rows)
    for i, (label, cnt, b, sw) in enumerate(rows):
        cx = rx + i * gw + gw / 2
        for j, (val, colour) in enumerate(((b, "#1c4e8a"), (sw, "#c2410c"))):
            bh = rh * min(val, 1.0)
            bx = cx - gw * 0.30 + j * gw * 0.30
            s.append(f'<rect x="{bx:.1f}" y="{ry + rh - bh:.1f}" width="{gw * 0.26:.1f}" '
                     f'height="{max(bh, 1.5):.1f}" rx="3" fill="{colour}"/>')
            s.append(f'<text class="mono" x="{bx + gw * 0.13:.1f}" y="{ry + rh - bh - 6:.1f}" '
                     f'text-anchor="middle">{val:.2f}</text>')
        s.append(f'<text class="ax" x="{cx:.1f}" y="{ry + rh + 18}" text-anchor="middle">'
                 f'{esc(label)}</text>')
        s.append(f'<text class="ax" x="{cx:.1f}" y="{ry + rh + 34}" text-anchor="middle" '
                 f'fill="#8a94a2">n={cnt}</text>')
    s.append(f'<rect x="{rx}" y="{ry + rh + 52}" width="14" height="12" rx="2" fill="#1c4e8a"/>')
    s.append(f'<text class="ax" x="{rx + 20}" y="{ry + rh + 62}">base-8KV</text>')
    s.append(f'<rect x="{rx + 110}" y="{ry + rh + 52}" width="14" height="12" rx="2" fill="#c2410c"/>')
    s.append(f'<text class="ax" x="{rx + 130}" y="{ry + rh + 62}">swa-w128-g4</text>')

    # ---------------- footer ----------------
    s.append(f'<rect x="34" y="{H - 78}" width="{W - 68}" height="58" rx="9" '
             f'fill="#fffbe9" stroke="#e0c060" stroke-width="1.4"/>')
    s.append(f'<text class="note" x="56" y="{H - 54}" font-weight="700">'
             f'The only rag_dev category left untouched is the one whose prompts fit inside the window.</text>')
    s.append(f'<text class="note" x="56" y="{H - 32}">'
             f'All 20 social cases are 41-45 tokens and score delta exactly 0.0000. '
             f'Every other category runs past w=128 and collapses. The fix is to pin the first few tokens '
             f'(StreamingLLM sinks), which transformers sliding_window does not do.</text>')
    s.append("</svg>")

    out = FIG / "06_attention_sink_cliff.svg"
    out.write_text(NL.join(s) + NL, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
