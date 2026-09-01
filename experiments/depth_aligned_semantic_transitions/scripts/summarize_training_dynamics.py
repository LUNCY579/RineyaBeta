"""Summarise DAST training-loss dynamics from completed (or training-complete) runs.

V2's gates are about optimization conflict, not generated-answer accuracy. This
script reads `metrics.jsonl` (per-step `loss/lm` and `loss/state`) and
`dev_summary.json` (final teacher-forced `L_state`) and prints seed-pooled
curves plus the three preregistered contrasts in `docs/PREREGISTRATION_V2.md`.

    python scripts/summarize_training_dynamics.py
    python scripts/summarize_training_dynamics.py --prefix v2_
    python scripts/summarize_training_dynamics.py --json out.json

Incomplete runs are named rather than omitted. Grouping is by `(arm, seed)` from
`run_manifest.json`, never from the directory name.

CLAIM BOUNDARY: this is not a V1 capability redo and not confirmatory until the
V2 preregistration is signed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# Preregistered V2-G1 margins. Kept here so the printed gate line cannot drift
# from PREREGISTRATION_V2.md without a code change that review will see.
MARGIN_L_STATE_ORDERED_MINUS_REVERSE = 0.005
MARGIN_L_STATE_ORDERED_MINUS_FINAL = 0.05

ORDERED = "state_ordered"
REVERSE = "state_reverse"
FINAL = "state_final_repeated"


def trapezoid_auc(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return area


def _step_key(row: dict[str, Any]) -> int | None:
    for key in ("optimizer_step", "global_step"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def load_metrics_curve(path: Path) -> dict[str, Any]:
    """Collapse metrics.jsonl to a strictly increasing step series."""

    by_step: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = _step_key(row)
            if step is None:
                continue
            by_step[step] = row
    steps = sorted(by_step)
    lm = [float(by_step[s]["loss/lm"]) for s in steps if "loss/lm" in by_step[s]]
    state = [float(by_step[s]["loss/state"]) for s in steps if "loss/state" in by_step[s]]
    lm_steps = [s for s in steps if "loss/lm" in by_step[s]]
    state_steps = [s for s in steps if "loss/state" in by_step[s]]
    return {
        "n_log_rows": len(by_step),
        "lm_auc": trapezoid_auc([float(s) for s in lm_steps], lm),
        "state_auc": trapezoid_auc([float(s) for s in state_steps], state),
        "final_lm": lm[-1] if lm else None,
        "final_state_train": state[-1] if state else None,
        "last_step": steps[-1] if steps else None,
    }


def identify_run(directory: Path) -> tuple[str, str] | None:
    try:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        return str(manifest["arm"]), str(manifest["seed"])
    except Exception:  # noqa: BLE001
        return None


def load_run(directory: Path) -> dict[str, Any] | None:
    metrics_path = directory / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    payload = load_metrics_curve(metrics_path)
    summary_path = directory / "dev_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["l_state"] = summary.get("loss/state")
        payload["lm_loss"] = summary.get("loss/lm")
        payload["ppl"] = summary.get("lm/perplexity")
        payload["has_dev_summary"] = True
    else:
        payload["l_state"] = payload["final_state_train"]
        payload["lm_loss"] = payload["final_lm"]
        payload["ppl"] = None
        payload["has_dev_summary"] = False
    return payload


def collect_runs(runs_dir: Path, prefixes: list[str]) -> tuple[dict[tuple[str, str], dict], list[str]]:
    found: dict[tuple[str, str], dict] = {}
    pending: list[str] = []
    candidates: list[Path] = []
    for prefix in prefixes:
        candidates.extend(sorted(runs_dir.glob(f"{prefix}*")))
    for directory in sorted(set(candidates)):
        if not directory.is_dir():
            continue
        loaded = load_run(directory)
        if loaded is None:
            pending.append(directory.name)
            continue
        identity = identify_run(directory)
        if identity is None:
            pending.append(directory.name)
            continue
        arm, seed = identity
        found[(arm, seed)] = {**loaded, "dir": directory.name}
    return found, pending


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def seed_diffs(
    found: dict[tuple[str, str], dict],
    left_arm: str,
    right_arm: str,
    key: str,
) -> list[tuple[str, float]]:
    seeds = sorted({seed for arm, seed in found if arm in {left_arm, right_arm}}, key=int)
    out: list[tuple[str, float]] = []
    for seed in seeds:
        left = found.get((left_arm, seed))
        right = found.get((right_arm, seed))
        if not (left and right):
            continue
        lv, rv = left.get(key), right.get(key)
        if lv is None or rv is None:
            continue
        out.append((seed, float(lv) - float(rv)))
    return out


def evaluate_v2_gates(found: dict[tuple[str, str], dict]) -> dict[str, Any]:
    c1 = seed_diffs(found, ORDERED, REVERSE, "l_state")
    c2 = seed_diffs(found, ORDERED, FINAL, "l_state")
    c3 = seed_diffs(found, ORDERED, REVERSE, "lm_auc")

    def _gate(diffs: list[tuple[str, float]], margin: float, require_margin: bool) -> dict[str, Any]:
        if not diffs:
            return {"available": False, "pass": False, "mean": None, "n": 0, "same_sign": False}
        mean = sum(d for _, d in diffs) / len(diffs)
        same_sign = all(d > 0 for _, d in diffs)
        passed = same_sign and (mean >= margin if require_margin else True)
        return {
            "available": True,
            "pass": passed,
            "mean": mean,
            "n": len(diffs),
            "same_sign": same_sign,
            "per_seed": {seed: diff for seed, diff in diffs},
        }

    g1_c1 = _gate(c1, MARGIN_L_STATE_ORDERED_MINUS_REVERSE, True)
    g1_c2 = _gate(c2, MARGIN_L_STATE_ORDERED_MINUS_FINAL, True)
    g2_c3 = _gate(c3, 0.0, False)
    return {
        "C1": g1_c1,
        "C2": g1_c2,
        "C3": g2_c3,
        "V2_G1": bool(g1_c1["available"] and g1_c2["available"] and g1_c1["pass"] and g1_c2["pass"]),
        "V2_G2": bool(g2_c3["available"] and g2_c3["pass"]),
    }


def render_report(found: dict[tuple[str, str], dict], pending: list[str], prefixes: list[str]) -> str:
    lines: list[str] = []
    lines.append(
        f"=== {'|'.join(prefixes)} : {len(found)} run(s) with metrics.jsonl, {len(pending)} incomplete ==="
    )
    if pending:
        lines.append("    incomplete: " + ", ".join(pending))
    if not found:
        lines.append("no runs with metrics.jsonl matching those prefixes")
        lines.append("CLAIM BOUNDARY: V2 is not a V1 capability redo. TEST is sealed.")
        return "\n".join(lines) + "\n"

    arms = sorted({arm for arm, _ in found})
    seeds = sorted({seed for _, seed in found}, key=int)

    lines.append("")
    header = f"{'arm':24s}" + "".join(f"{'s'+s:>10}" for s in seeds) + f"{'mean':>10}{'n':>4}"
    lines.append("final L_state (dev_summary, else last train loss/state)")
    lines.append(header)
    lines.append("-" * len(header))
    for arm in arms:
        cells = ""
        values: list[float] = []
        for seed in seeds:
            entry = found.get((arm, seed))
            if entry and entry.get("l_state") is not None:
                values.append(float(entry["l_state"]))
                cells += f"{entry['l_state']:>10.4f}"
            else:
                cells += f"{'-':>10}"
        mean = sum(values) / len(values) if values else None
        lines.append(f"{arm:24s}{cells}{_fmt(mean):>10}{len(values):>4d}")

    lines.append("")
    header2 = f"{'arm':24s}" + "".join(f"{'s'+s:>10}" for s in seeds) + f"{'mean':>10}"
    lines.append("training loss/lm trapezoidal AUC")
    lines.append(header2)
    lines.append("-" * len(header2))
    for arm in arms:
        cells = ""
        values = []
        for seed in seeds:
            entry = found.get((arm, seed))
            if entry and entry.get("lm_auc") is not None:
                values.append(float(entry["lm_auc"]))
                cells += f"{entry['lm_auc']:>10.4f}"
            else:
                cells += f"{'-':>10}"
        mean = sum(values) / len(values) if values else None
        lines.append(f"{arm:24s}{cells}{_fmt(mean):>10}")

    lines.append("")
    lines.append("final logged training loss/lm")
    lines.append(header2)
    lines.append("-" * len(header2))
    for arm in arms:
        cells = ""
        values = []
        for seed in seeds:
            entry = found.get((arm, seed))
            if entry and entry.get("final_lm") is not None:
                values.append(float(entry["final_lm"]))
                cells += f"{entry['final_lm']:>10.4f}"
            else:
                cells += f"{'-':>10}"
        mean = sum(values) / len(values) if values else None
        lines.append(f"{arm:24s}{cells}{_fmt(mean):>10}")

    gates = evaluate_v2_gates(found)
    lines.append("")
    lines.append("=== preregistered V2 contrasts (see PREREGISTRATION_V2.md) ===")
    for cid, label, margin in (
        ("C1", "ordered − reverse L_state", MARGIN_L_STATE_ORDERED_MINUS_REVERSE),
        ("C2", "ordered − final_repeated L_state", MARGIN_L_STATE_ORDERED_MINUS_FINAL),
        ("C3", "ordered − reverse loss/lm AUC", None),
    ):
        row = gates[cid]
        if not row["available"]:
            lines.append(f"{cid} {label}: unavailable")
            continue
        margin_txt = f" margin {margin}" if margin is not None else " (sign only)"
        verdict = "PASS" if row["pass"] else "FAIL"
        per = ", ".join(f"s{s} {d:+.4f}" for s, d in sorted(row["per_seed"].items(), key=lambda kv: int(kv[0])))
        lines.append(
            f"{cid} {label}: mean {row['mean']:+.4f} n={row['n']} same_sign={row['same_sign']}"
            f"{margin_txt} -> {verdict} ({per})"
        )
    lines.append(f"V2-G1 (C1 and C2): {'PASS' if gates['V2_G1'] else 'FAIL'}")
    lines.append(f"V2-G2 (C3): {'PASS' if gates['V2_G2'] else 'FAIL'}")
    lines.append("")
    lines.append("CLAIM BOUNDARY: not a V1 capability redo. Unsigned V2 preregistration")
    lines.append("cannot open TEST. Accuracy, if present, is a consistency check, not a gate.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, runs_dir: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise DAST training-loss dynamics")
    parser.add_argument("--prefix", default="v2_,g2_budget_,g3_")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)

    prefixes = [p.strip() for p in args.prefix.split(",") if p.strip()]
    found, pending = collect_runs(runs_dir or RUNS, prefixes)
    report = render_report(found, pending, prefixes)
    print(report, end="")

    if args.json_out:
        payload = {
            "prefix": args.prefix,
            "runs": {f"{a}_s{s}": {k: v for k, v in e.items() if k != "dir"} for (a, s), e in found.items()},
            "incomplete": pending,
            "gates": evaluate_v2_gates(found) if found else {},
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
