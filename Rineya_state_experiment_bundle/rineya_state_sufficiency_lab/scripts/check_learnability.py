"""Gate G9 -- learnability / headroom.

The untouched checkpoint scores near zero on this task. If the FULL arm also
lands near zero, an A-vs-C null result is uninterpretable: "sequence-free
training is just as good" and "neither representation worked" produce the same
number. This gate stops the experiment before that ambiguity is baked in.

The thresholds must be declared in docs/PREREGISTRATION.md section 8b BEFORE the
number is seen; passing them in as environment variables at run time is how they
stay pre-committed rather than chosen after the fact.
"""

from __future__ import annotations

import json
import os
import sys


def _strict(path: str, protocol: str = "full_direct") -> float:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    summary = report.get("protocols", {}).get(protocol)
    if summary is None:
        raise SystemExit(f"GATE G9: {path} has no '{protocol}' protocol to check")
    return summary["strict"]["accuracy"] or 0.0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        raise SystemExit("usage: check_learnability.py <full_eval_report.json> <base_eval_report.json>")
    floor_pct = float(os.environ.get("LEARNABILITY_FLOOR_PCT", "25"))
    min_lift_pp = float(os.environ.get("LEARNABILITY_MIN_LIFT_PP", "15"))

    full = _strict(argv[1])
    base = _strict(argv[2])
    lift_pp = 100 * (full - base)

    print(f"FULL full_direct strict = {100 * full:.2f}%")
    print(f"base full_direct strict = {100 * base:.2f}%")
    print(f"lift                    = {lift_pp:+.2f} pp")
    print(f"declared floor          = {floor_pct:.2f}% absolute, {min_lift_pp:.2f} pp over base")

    problems = []
    if 100 * full < floor_pct:
        problems.append(f"absolute accuracy {100 * full:.2f}% is below the {floor_pct:.2f}% floor")
    if lift_pp < min_lift_pp:
        problems.append(f"lift {lift_pp:+.2f} pp is below the {min_lift_pp:.2f} pp minimum")

    if problems:
        print("\nGATE G9 FAILED: " + "; ".join(problems), file=sys.stderr)
        print(
            "Both arms would sit near the floor, and an A-vs-C null would be uninterpretable.\n"
            "Stop here. Remedies, in order of preference:\n"
            "  (a) use a healthier base checkpoint;\n"
            "  (b) raise LoRA capacity (r 32-64) and/or train for more epochs;\n"
            "  (c) restrict to the categories FULL can learn, and report that restriction.\n"
            "See docs/READINESS_ASSESSMENT.md section 7b.",
            file=sys.stderr,
        )
        return 1
    print("\nGATE G9 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
