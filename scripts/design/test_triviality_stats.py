#!/usr/bin/env python3
"""Validate the triviality control's exact statistics against closed forms, against
R's binom.test, and against this programme's OWN published p-values.

There is no scipy in this environment, so exact McNemar and Clopper-Pearson are
hand-written in analyse_triviality_control.py. A hand-written exact test that is
never checked against an independent implementation is how a p-value becomes
decorative -- and this programme has already withdrawn one set of p-values for
modelling paired arms as independent groups.

Run:  python scripts/design/test_triviality_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.design.analyse_triviality_control import (  # noqa: E402
    clopper_pearson, mcnemar_exact, paired_diff_ci)

FAILED: list[str] = []


def check(name: str, got, want, tol: float = 1e-9) -> None:
    ok = abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAILED.append(name)


def check_true(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


print("\n1. exact McNemar against this programme's published values")
check("6/0 -> 0.03125 (prereg 5's minimum detectable)", mcnemar_exact(6, 0), 0.03125)
check("164/0 -> 2^-163 (paper: 8.6e-50)", mcnemar_exact(164, 0), 2.0 ** -163, 1e-6)
check("117/83 -> 0.0194 (paper 4.10)", round(mcnemar_exact(117, 83), 4), 0.0194, 1e-3)
check("0/0 -> 1.0", mcnemar_exact(0, 0), 1.0)
check("symmetric 10/10 -> 1.0", mcnemar_exact(10, 10), 1.0)
check_true("5/0 does NOT reach 0.05", mcnemar_exact(5, 0) > 0.05,
           f"p={mcnemar_exact(5, 0)}")
check_true("exact test is discrete and non-monotone in n_disc",
           mcnemar_exact(8, 1) > mcnemar_exact(6, 0),
           "noted in prereg 5 so a marginal cell is not misread")

print("\n2. Clopper-Pearson against the closed form and against R's binom.test")
for n in (100, 300):
    lo, hi = clopper_pearson(n, n)
    check(f"CP({n},{n}) lower == 0.025^(1/n)", lo, 0.025 ** (1.0 / n), 1e-6)
    check(f"CP({n},{n}) upper == 1", hi, 1.0)
lo, hi = clopper_pearson(0, 50)
check("CP(0,50) lower == 0", lo, 0.0)
check("CP(0,50) upper == 1-0.025^(1/50)", hi, 1 - 0.025 ** (1 / 50), 1e-6)
lo, hi = clopper_pearson(5, 20)
check("CP(5,20) lower == R's 0.08657", lo, 0.08657, 1e-3)
check("CP(5,20) upper == R's 0.49104", hi, 0.49104, 1e-3)
lo, hi = clopper_pearson(3, 10)
check("CP(3,10) lower == R's 0.06674", lo, 0.06674, 1e-3)
check("CP(3,10) upper == R's 0.65245", hi, 0.65245, 1e-3)

print("\n3. paired difference CI")
lo, hi = paired_diff_ci(0, 0, 200)
check("no discordance at n=200 -> rule of three", hi, 3.0 / 200)
check_true("interval is symmetric with no discordance", abs(lo + hi) < 1e-12)
lo, hi = paired_diff_ci(20, 20, 144)
check_true("balanced discordance brackets zero", lo < 0 < hi, f"[{lo:.4f},{hi:.4f}]")
lo, hi = paired_diff_ci(164, 0, 200)
check_true("all-one-way interval excludes zero", lo > 0, f"[{lo:.4f},{hi:.4f}]")

print("\n4. the TRIVIAL branch is REACHABLE at the preregistered n (prereg 5)")
for n, want in ((144, True), (72, True)):
    best = -1
    for nd in range(0, n + 1, 2):
        lo, hi = paired_diff_ci(nd // 2, nd // 2, n)
        if hi <= 0.10 and lo >= -0.10:
            best = nd
    check_true(f"n={n}: +/-10pp survives up to {best} balanced discordant pairs",
               (best > 0) == want, f"{best / n:.1%} of cases")

print("\n5. the R statistic's guard rails (prereg 6.5)")
# R = (L-N)/(A-N). Two negatives must not silently make a positive.
A, N, L = 0.60, 0.65, 0.62
check_true("A < N would give a spuriously positive R without the gate",
           (L - N) / (A - N) > 0,
           f"R={(L - N) / (A - N):.3f} while significance LOSES to the floor -- "
           f"which is why prereg 6.5 requires A - N >= 0.40 AND A > N")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all triviality-control statistics tests pass")
