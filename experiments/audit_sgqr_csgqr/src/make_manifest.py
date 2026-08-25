#!/usr/bin/env python3
"""Assemble RUN_MANIFEST.json: one row per run, with everything needed to repeat it.

The brief asks every new experiment to carry a run manifest with the git commit,
model, retired layers, quantization configuration, significance configuration,
gate type, threshold, execution budget, seed, datasets, evaluation settings and
environment. Each of this audit's result files already embeds its own
configuration and environment block; this script indexes them into one file so a
reader does not have to know which key lives in which artifact, and hashes every
artifact so the index cannot drift from what it describes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUD = ROOT / "experiments/audit_sgqr_csgqr"
RES = AUD / "results"


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a],
                          capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dig(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def main() -> int:
    runs = []
    for p in sorted(RES.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        env = d.get("environment") or {}
        runs.append({
            "artifact": p.relative_to(ROOT).as_posix(),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "block": d.get("block") or d.get("audit") or d.get("campaign"),
            "split": d.get("split") or d.get("dataset"),
            "split_range": d.get("split_range") or d.get("range")
            or d.get("calibration_split"),
            "prompts": d.get("prompts") or d.get("calibration_prompts")
            or d.get("n_prompts"),
            "positions": d.get("reply_positions") or d.get("positions")
            or d.get("n_positions"),
            "retired_layers": d.get("retired_layers"),
            "n_groups_per_layer": d.get("n_groups_per_layer") or d.get("n_groups"),
            "quantization": d.get("spec") or d.get("bits"),
            "significance_source": d.get("significance_source")
            or d.get("ranking_from"),
            "probes": d.get("probes"),
            "budgets": d.get("budgets"),
            "rung_groups": d.get("rung_groups"),
            "budget_target_group_tokens": d.get("budget_target_group_tokens"),
            "seed": d.get("seed") or d.get("seeds"),
            "n_boot": d.get("n_boot"),
            "max_len": d.get("max_len"),
            "unit_of_analysis": d.get("unit_of_analysis"),
            "matched_compute": d.get("matched_compute"),
            "git_sha": env.get("git_sha"),
            "git_branch": env.get("git_branch"),
            "git_dirty": env.get("git_dirty"),
            "torch": env.get("torch"),
            "transformers": env.get("transformers"),
            "gpu": dig(env, "gpu", "name"),
            "gpu_total_vram_mib": dig(env, "gpu", "total_vram_mib"),
            "gpu_used_by_others_mib": dig(env, "gpu", "used_by_others_mib"),
            "elapsed_s": d.get("elapsed_s"),
        })

    other = []
    for pat in ("*.jsonl", "*.csv", "*.pt", "*.log"):
        for p in sorted(RES.glob(pat)):
            other.append({"artifact": p.relative_to(ROOT).as_posix(),
                          "sha256": sha256(p), "bytes": p.stat().st_size})

    manifest = {
        "audit": "sgqr_csgqr_dual",
        "audit_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "audit_head": git("rev-parse", "HEAD"),
        "base_commit": git("rev-parse", "d527c19"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "model": "artifacts/rineya-core-en32k-28L",
        "retired_layers": [13, 14, 24, 25],
        "layer_selection_rule": ("four lowest deletion damage in the committed "
                                 "experiments/csgqr/results/layer_scan.json"),
        "splits": {"calib": [0, 96], "val": [96, 224], "test": [224, 510],
                   "rule": ("significance and probes fitted on calib; ridge "
                            "strength selected on val; nothing selected on test")},
        "sealed_sets_read": False,
        "n_json_runs": len(runs),
        "runs": runs,
        "other_artifacts": other,
    }
    (AUD / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                           encoding="utf-8")
    print("indexed %d JSON runs and %d other artifacts -> RUN_MANIFEST.json"
          % (len(runs), len(other)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
