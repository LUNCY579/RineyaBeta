"""B2-T verdict. Implements the FROZEN analysis plan, amendment 2.

Protocol: docs/PREREGISTRATION_B2_TRAJECTORY.md. Section 10 (amendment 2) governs,
then section 9, then section 5.

Primary test
------------
The unit of replication is the INITIALISATION, not the step. For each init seed k,
all three arms share one initialisation and one batch order, so the paired per-step
difference is attributable to the arm. Compute

    D_k(control) = mean over EARLY (global steps 1-120) of
                   lora_only_cosine_ordered(t) - lora_only_cosine_control(t)

and test {D_k} against zero with a one-sample t on K-1 degrees of freedom.

Why not a within-run bootstrap: any component of an arm mean that is CONSTANT across
the window -- the shared initialisation draw, the weight path -- contributes exactly
zero to a bootstrap over steps within one run. Amendment 1's block bootstrap could
not see it, and a nuisance of the size this programme has already measured turns the
false-positive rate for SUPPORTED from ~1% into 7-12%. Replicating the
initialisation is the only way to put that variance inside the error term.

Fail-safe
---------
REFUTED is the verdict that closes the DAST programme, and a naive
"no interval excluded zero" rule makes it the DEFAULT output on absent, NaN or
truncated data. This script therefore REFUSES to emit any verdict unless the
preregistered minimum coverage is present. A missing measurement must never read as
evidence of no effect.

Usage
-----
    python scripts/gradient_trajectory_analyse.py --out artifacts/b2t_verdict.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "state_ordered"
CONTROLS = ("state_reverse", "state_final_repeated")
ARMS = (REFERENCE,) + CONTROLS
ACCUMULATION = 8  # grad_accum: 8 global steps = 1 optimizer step = one block

WINDOWS = {"EARLY": (1, 120), "MID": (121, 400), "LATE": (1601, 2000)}
REQUIRED_EARLY_BLOCKS = 15  # 120 global steps / 8
MIN_SEEDS = 5
# Two-sided t critical values at alpha=0.05, indexed by degrees of freedom.
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
# lora_only_cosine from the step-250 snapshot, artifacts/b2_verdict_s29.json.
SNAPSHOT = {"state_ordered": -0.00623, "state_reverse": +0.00475,
            "state_final_repeated": -0.00364}

RUN_PATTERN = re.compile(r"^diag_b2traj_s(\d+)_(state_\w+)$")


def load_trajectory(path: Path, metric: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("measured"):
                continue
            value = row.get(metric)
            if value is None or value != value:  # skip None and NaN
                continue
            values[int(row["global_step"])] = float(value)
    return values


def discover(runs_dir: Path, metric: str) -> tuple[dict, dict]:
    trajectories: dict[tuple[int, str], dict[int, float]] = {}
    digests: dict[tuple[int, str], str] = {}
    for directory in sorted(runs_dir.iterdir()):
        match = RUN_PATTERN.match(directory.name)
        if not match or not (directory / "grad_trajectory.jsonl").exists():
            continue
        seed, arm = int(match.group(1)), match.group(2)
        if arm not in ARMS:
            continue
        trajectories[(seed, arm)] = load_trajectory(
            directory / "grad_trajectory.jsonl", metric
        )
        summary = directory / "grad_trajectory_summary.json"
        if summary.exists():
            digests[(seed, arm)] = json.loads(summary.read_text(encoding="utf-8")).get(
                "init_trainable_state_sha256", ""
            )
    return trajectories, digests


def blocks_in(steps, low: int, high: int) -> int:
    return len({(step - 1) // ACCUMULATION for step in steps if low <= step <= high})


def one_sample_t(values: list[float]) -> dict:
    count = len(values)
    mean = statistics.mean(values)
    if count < 2:
        return {"n": count, "mean": mean, "sd": float("nan"), "se": float("nan"),
                "t": float("nan"), "df": count - 1, "t_critical": float("nan"),
                "significant": False, "ci_low": float("nan"), "ci_high": float("nan")}
    deviation = statistics.stdev(values)
    error = deviation / math.sqrt(count)
    df = count - 1
    critical = T_CRITICAL.get(df, 1.96)
    return {
        "n": count, "mean": mean, "sd": deviation, "se": error,
        "t": mean / error if error > 0 else float("nan"), "df": df,
        "t_critical": critical,
        "significant": abs(mean) > critical * error if error > 0 else False,
        "ci_low": mean - critical * error, "ci_high": mean + critical * error,
    }


def block_bootstrap(paired: dict[int, float], *, samples: int = 10000, seed: int = 1701):
    blocks: dict[int, list[float]] = {}
    for step, value in paired.items():
        blocks.setdefault((step - 1) // ACCUMULATION, []).append(value)
    keys = sorted(blocks)
    if len(keys) < 2:
        raise ValueError(
            f"block bootstrap needs at least 2 blocks, got {len(keys)}; refusing to "
            "return an interval that would read as a measurement"
        )
    totals = torch.tensor([sum(blocks[k]) for k in keys], dtype=torch.float64)
    counts = torch.tensor([len(blocks[k]) for k in keys], dtype=torch.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    draws = torch.randint(0, len(keys), (samples, len(keys)), generator=generator)
    means = totals[draws].sum(dim=1) / counts[draws].sum(dim=1)
    ordered = means.sort().values
    return (float(totals.sum() / counts.sum()),
            float(ordered[int(0.025 * samples)]),
            float(ordered[int(0.975 * samples)]),
            len(keys))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="lora_only_cosine")
    parser.add_argument("--min-seeds", type=int, default=MIN_SEEDS)
    parser.add_argument(
        "--allow-underpowered",
        action="store_true",
        help="report per-seed numbers without a verdict when coverage is short",
    )
    args = parser.parse_args()

    trajectories, digests = discover(Path(args.runs), args.metric)
    if not trajectories:
        raise SystemExit("no B2-T trajectories found; nothing to analyse")

    seeds = sorted({seed for seed, _ in trajectories})
    complete_seeds = []
    coverage = {}
    for seed in seeds:
        per_arm = {}
        for arm in ARMS:
            steps = trajectories.get((seed, arm), {})
            per_arm[arm] = {
                "measured": len(steps),
                "early_blocks": blocks_in(steps, *WINDOWS["EARLY"]),
                "max_step": max(steps, default=0),
            }
        shared = (
            set.intersection(*(set(trajectories[(seed, a)]) for a in ARMS))
            if all((seed, a) in trajectories for a in ARMS) else set()
        )
        early_blocks_shared = blocks_in(shared, *WINDOWS["EARLY"])
        matched_init = len({digests.get((seed, a), f"missing-{a}") for a in ARMS}) == 1
        coverage[seed] = {
            "per_arm": per_arm,
            "early_blocks_shared": early_blocks_shared,
            "init_digests_match": matched_init,
            "usable": early_blocks_shared >= REQUIRED_EARLY_BLOCKS and matched_init,
        }
        if coverage[seed]["usable"]:
            complete_seeds.append(seed)

    print("=== coverage ===")
    print(f"  {'seed':>6} {'shared EARLY blocks':>20} {'init matched':>13} {'usable':>7}")
    for seed in seeds:
        row = coverage[seed]
        print(f"  {seed:>6} {row['early_blocks_shared']:>20} "
              f"{str(row['init_digests_match']):>13} {str(row['usable']):>7}")
    print(f"\n  usable seeds: {complete_seeds} (need >= {args.min_seeds}, "
          f"{REQUIRED_EARLY_BLOCKS} shared EARLY blocks and matched init per seed)")

    # --- FAIL-SAFE ------------------------------------------------------------
    # A missing measurement must never read as evidence of no effect.
    if len(complete_seeds) < args.min_seeds:
        message = (
            f"INSUFFICIENT COVERAGE: {len(complete_seeds)} usable seeds, need "
            f"{args.min_seeds}. NO VERDICT is emitted. This is not a null result and "
            "must not be recorded as one -- 'H2-T REFUTED' closes the DAST programme "
            "and cannot be allowed to be the default output of absent data."
        )
        payload = {
            "metric": args.metric,
            "coverage": coverage,
            "usable_seeds": complete_seeds,
            "verdict": None,
            "status": "INSUFFICIENT_COVERAGE",
            "message": message,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n{message}\nwrote {args.out}")
        return 0 if args.allow_underpowered else 3

    # --- PRIMARY: one-sample t over per-seed paired differences ---------------
    print("\n=== PRIMARY: per-seed EARLY paired difference, then one-sample t ===")
    per_seed: dict[str, list[float]] = {control: [] for control in CONTROLS}
    low, high = WINDOWS["EARLY"]
    print(f"  {'seed':>6} " + "".join(f"{'ordered-' + c.replace('state_', ''):>26s}"
                                      for c in CONTROLS))
    for seed in complete_seeds:
        cells = []
        shared = set.intersection(*(set(trajectories[(seed, a)]) for a in ARMS))
        for control in CONTROLS:
            paired = [
                trajectories[(seed, REFERENCE)][step] - trajectories[(seed, control)][step]
                for step in sorted(shared) if low <= step <= high
            ]
            value = statistics.mean(paired)
            per_seed[control].append(value)
            cells.append(value)
        print(f"  {seed:>6} " + "".join(f"{value:>+26.6f}" for value in cells))

    primary = {}
    print(f"\n  {'contrast':34s} {'mean D':>11} {'sd':>10} {'t':>8} {'df':>4} "
          f"{'t_crit':>7} {'95% CI':>26} {'signif':>7}")
    for control in CONTROLS:
        test = one_sample_t(per_seed[control])
        primary[f"{REFERENCE} - {control}"] = test
        print(f"  {REFERENCE + ' - ' + control:34s} {test['mean']:>+11.6f} "
              f"{test['sd']:>10.6f} {test['t']:>+8.2f} {test['df']:>4} "
              f"{test['t_critical']:>7.3f} [{test['ci_low']:>+.6f},{test['ci_high']:>+.6f}] "
              f"{'yes' if test['significant'] else 'no':>7}")

    supported = all(
        primary[f"{REFERENCE} - {c}"]["mean"] < 0
        and primary[f"{REFERENCE} - {c}"]["significant"]
        for c in CONTROLS
    )
    refuted = all(not primary[f"{REFERENCE} - {c}"]["significant"] for c in CONTROLS)

    # --- SECONDARY, descriptive only -----------------------------------------
    secondary: dict = {}
    full_seeds = [
        seed for seed in seeds
        if all(
            blocks_in(trajectories.get((seed, a), {}), *WINDOWS["LATE"]) >= 2 for a in ARMS
        )
    ]
    if full_seeds:
        seed = full_seeds[0]
        print(f"\n=== SECONDARY (descriptive): full-trajectory windows, seed {seed} ===")
        shared = set.intersection(*(set(trajectories[(seed, a)]) for a in ARMS))
        for name, (low_w, high_w) in WINDOWS.items():
            secondary[name] = {}
            for control in CONTROLS:
                paired = {
                    step: trajectories[(seed, REFERENCE)][step]
                    - trajectories[(seed, control)][step]
                    for step in shared if low_w <= step <= high_w
                }
                try:
                    mean, ci_low, ci_high, blocks = block_bootstrap(paired)
                except ValueError as error:
                    secondary[name][f"{REFERENCE} - {control}"] = {"error": str(error)}
                    print(f"  {name:6s} {control:22s} skipped: {error}")
                    continue
                secondary[name][f"{REFERENCE} - {control}"] = {
                    "paired_mean": mean, "ci_low": ci_low, "ci_high": ci_high,
                    "n_blocks": blocks,
                }
                print(f"  {name:6s} ordered-{control.replace('state_',''):16s} "
                      f"{mean:>+11.6f} [{ci_low:>+.6f},{ci_high:>+.6f}] "
                      f"blocks {blocks}")
        print("\n  snapshot check (LATE vs the step-250 snapshot; sign agreement is the "
              "meaningful test -- different optimizer path)")
        snapshot = {}
        for arm in ARMS:
            values = [v for s, v in trajectories[(seed, arm)].items()
                      if WINDOWS["LATE"][0] <= s <= WINDOWS["LATE"][1]]
            late = statistics.mean(values) if values else float("nan")
            agrees = (late < 0) == (SNAPSHOT[arm] < 0) if late == late else False
            snapshot[arm] = {"b2t_late": late, "snapshot": SNAPSHOT[arm],
                             "sign_agrees": agrees}
            print(f"    {arm:22s} {late:>+12.6f} vs {SNAPSHOT[arm]:>+12.6f}  "
                  f"{'agrees' if agrees else 'DISAGREES'}")
        secondary["snapshot_check"] = snapshot

    if supported:
        verdict = (
            "H2-T SUPPORTED: across independent initialisations, ordered's early-training "
            "paired cosine is more negative than BOTH controls. H2 revives as an "
            "EARLY-TRAINING phenomenon. Per the frozen consequences this licenses a "
            "lambda-schedule or gradient-surgery test (B3); it does NOT by itself reopen "
            "DAST training or unpark the V2 matrix."
        )
    elif refuted:
        verdict = (
            "H2-T REFUTED: with the initialisation replicated, neither contrast differs "
            "from zero. The gradient-conflict story is closed at trajectory resolution as "
            "well as at convergence, and the last open branch of the diagnosis tree shuts."
        )
    else:
        verdict = (
            "H2-T AMBIGUOUS: one contrast is significant but not the preregistered pattern "
            "(ordered more negative than BOTH controls). Recorded as ambiguous; no "
            "successor is licensed and the V2 matrix stays parked."
        )
    print(f"\nVERDICT: {verdict}")

    payload = {
        "metric": args.metric,
        "protocol": "docs/PREREGISTRATION_B2_TRAJECTORY.md (section 10 governs)",
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "accumulation_block": ACCUMULATION,
        "coverage": coverage,
        "usable_seeds": complete_seeds,
        "per_seed_early_paired_difference": per_seed,
        "primary_one_sample_t": primary,
        "secondary_descriptive": secondary,
        "h2t_supported": supported,
        "h2t_refuted": refuted,
        "status": "COMPLETE",
        "verdict": verdict,
        "claim_boundary": (
            "Diagnosis only, DEV only, TEST sealed. No capability endpoint: these runs "
            "disabled generation and their accuracy must not be reported."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
