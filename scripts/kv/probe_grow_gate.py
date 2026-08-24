#!/usr/bin/env python3
"""O4, on CPU: why does the grow multiplier matter at long spans and not short ones?

THE OBSERVATION. `grow=2.0` and `grow=5.0` are statistically identical at d=4 and d=5
(exact McNemar p = 1.0000 both) and differ sharply at d=7: 0.889 against 0.611,
p = 0.0020. The explanation offered in FINDINGS_P2.md is that a stricter gate excludes
weak payload positions so growth cannot traverse a longer span. That was ASSERTED from
the design, never demonstrated.

WHY THIS CAN BE SETTLED WITHOUT THE GPU. `grow_spans` is a pure function of a score
tensor. The question is not what the model produces but what the function DOES with a
score profile of a given shape, and that shape is known from P2.2: a query-shaped head
that scores high, a payload that scores well above filler but far below the head, and
filler either side. Feed that shape in at several span lengths and multipliers and the
behaviour is fully determined.

WHAT IS MEASURED. For each (span length, multiplier): how many payload positions clear
the noise gate, and how many end up carrying the head's score after growth. If the
asserted explanation is right, coverage should fall as the multiplier rises AND the
effect should be larger at longer spans -- a genuine interaction, not a main effect.
If coverage is identical across multipliers, the explanation is wrong and O4 needs a
different candidate.

Usage:
    python scripts/kv/probe_grow_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kv.significance import grow_spans  # noqa: E402


def profile(n: int, start: int, span: int, head: float, payload: float,
            filler: float) -> torch.Tensor:
    """A query-shaped head followed by a weaker payload run, in filler.

    Shaped after what P2.2 measured on the real cache: the framing scores high because
    it *is* the query lexically, and the payload it introduces scores well below it.
    """
    s = torch.full((1, 1, n), filler)
    s[0, 0, start] = head
    s[0, 0, start + 1:start + 1 + span] = payload
    return s


def main() -> None:
    n, start = 400, 120
    head, payload, filler = 1.0, 0.15, 0.01
    spans = [4, 5, 7, 9, 12, 20]
    mults = [2.0, 3.0, 5.0, 8.0]

    print("O4 probe -- payload coverage after growth, by span length and multiplier\n")
    print(f"  profile: head {head}, payload {payload}, filler {filler}"
          f"  (median of the stream is the filler level)")
    print(f"  coverage = payload positions carrying the head's score after growth\n")

    print(f"{'span':>5} | " + " ".join(f"{'mult ' + str(m):>10}" for m in mults))
    print("-" * (7 + 11 * len(mults)))
    table = {}
    for span in spans:
        s = profile(n, start, span, head, payload, filler)
        row = []
        for m in mults:
            g = grow_spans(s, m, 64)[0, 0]
            body = g[start + 1:start + 1 + span]
            cov = int((body >= head - 1e-6).sum())
            table[(span, m)] = cov / span
            row.append(f"{cov:>3}/{span:<3} {cov/span:>4.0%}")
        print(f"{span:>5} | " + " ".join(f"{r:>10}" for r in row))

    print("\nIs the multiplier's effect LARGER at longer spans (an interaction),")
    print("or the same everywhere (a main effect that cannot explain O4)?\n")
    print(f"{'span':>5} {'cov@2.0':>9} {'cov@5.0':>9} {'drop':>8}")
    for span in spans:
        a, b = table[(span, 2.0)], table[(span, 5.0)]
        print(f"{span:>5} {a:>8.0%} {b:>8.0%} {a - b:>+8.0%}")

    drops = [table[(s_, 2.0)] - table[(s_, 5.0)] for s_ in spans]
    short = sum(drops[:2]) / 2
    long_ = sum(drops[-2:]) / 2
    print(f"\n  mean drop at the two SHORTEST spans: {short:+.0%}")
    print(f"  mean drop at the two LONGEST  spans: {long_:+.0%}")
    if abs(long_) > abs(short) + 0.05:
        print("\n  -> INTERACTION present: the stricter gate costs more at longer spans,")
        print("     which is the mechanism FINDINGS_P2.md asserted for O4.")
    elif abs(short) < 0.05 and abs(long_) < 0.05:
        print("\n  -> NO effect at any span. The asserted explanation is WRONG and the")
        print("     multiplier must act through something other than gate coverage.")
    else:
        print("\n  -> Main effect without interaction: the gate costs the same at every")
        print("     span, so it cannot explain a difference that appears only at d=7.")

    print("\nCAVEAT. This is the score profile P2.2 measured, idealised: one head, a")
    print("uniform payload, uniform filler. Real streams are noisier and the payload is")
    print("not flat. It establishes what the FUNCTION does, not what the cache contains.")



def sweep_payload() -> None:
    """At what payload level does the multiplier START to matter?

    The first table shows the gate is inert on the profile P2.2 implies: with filler at
    0.01 and payload at 0.15, even mult=8 gates at 0.08 and every payload position
    clears it. So gate coverage cannot explain O4 on that profile.

    That leaves a sharper question. The multiplier demonstrably DOES matter at d=7 in
    the real cache (0.889 against 0.611, p = 0.0020), so if the mechanism is gate
    coverage then the real payload must sit much closer to the noise floor than the
    idealised profile assumes. This sweep finds where the boundary is, which turns a
    refuted explanation into a testable prediction about the real score distribution.
    """
    n, start, span = 400, 120, 7
    head, filler = 1.0, 0.01
    print(chr(10) * 2 + "At what payload level does the multiplier begin to bite? (span 7)" + chr(10))
    print(f"{'payload':>8} {'ratio':>7} | " + " ".join(f"{'mult ' + str(m):>9}" for m in (2.0, 3.0, 5.0, 8.0)))
    print("-" * 58)
    for payload in (0.20, 0.10, 0.05, 0.03, 0.02, 0.015, 0.012):
        s_ = profile(n, start, span, head, payload, filler)
        row = []
        for m in (2.0, 3.0, 5.0, 8.0):
            g = grow_spans(s_, m, 64)[0, 0]
            cov = int((g[start + 1:start + 1 + span] >= head - 1e-6).sum())
            row.append(f"{cov}/{span}")
        print(f"{payload:>8.3f} {payload/filler:>6.1f}x | " + " ".join(f"{r:>9}" for r in row))
    print(chr(10) + "  ratio = payload score divided by filler score. The gate sits at")
    print("  median x mult, and the median IS the filler level, so the multiplier")
    print("  only bites once the payload falls below mult x filler.")
    print(chr(10) + "  PREDICTION this makes about the real cache: if gate coverage is really")
    print("  the mechanism behind O4, then at d=7 the payload must score below ~5x the")
    print("  stream median. That is directly checkable by dumping real scores, and it")
    print("  is a much sharper claim than the one FINDINGS_P2.md asserted.")


if __name__ == "__main__":
    main()
