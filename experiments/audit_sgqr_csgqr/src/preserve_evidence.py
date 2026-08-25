#!/usr/bin/env python3
"""Evidence preservation pass 1: hash every SGQR/CSGQR artifact and establish
git provenance for it.

Read-only with respect to the historical record. Writes only into
experiments/audit_sgqr_csgqr/evidence/.

For every result artifact we record:
  * sha256 of the bytes on disk now
  * whether the working-tree bytes match the bytes in the commit that carries it
  * the commit that last modified it (author date, subject)
  * for each producing script named in the artifact's own manifest (or inferred),
    the commit that last modified that script, and whether that commit is
    LATER than the commit carrying the artifact.  A "yes" means the reported
    number was produced by code that has since changed.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/audit_sgqr_csgqr/evidence"
OUT.mkdir(parents=True, exist_ok=True)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_commit(path: str) -> dict:
    raw = git("log", "-1", "--format=%H%x1f%ad%x1f%s", "--date=iso-strict", "--", path)
    if not raw:
        return {"commit": None, "date": None, "subject": None, "tracked": False}
    h, d, s = raw.split("\x1f")
    return {"commit": h, "date": d, "subject": s, "tracked": True}


def commit_order() -> dict:
    """Topological index of every commit reachable from any ref, so we can ask
    'is commit A later than commit B' without relying on wall-clock dates."""
    order = {}
    for i, h in enumerate(git("rev-list", "--all", "--reverse").splitlines()):
        order[h] = i
    return order


TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".csv", ".txt", ".log",
                 ".yaml", ".yml", ".sh"}


def _norm(b: bytes, path: str) -> bytes:
    """Strip CR before hashing text files.

    ``.gitattributes`` sets ``text=auto eol=lf``, so the clean filter removes CR
    on the way into the index while the working copy may still carry CRLF. A raw
    byte comparison against the blob therefore reports every text file as
    diverged, which is an artefact of the comparison and not a fact about the
    repository. Normalising here makes the check mean what it is supposed to
    mean: has the CONTENT moved since the commit that carries it.
    """
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        return b.replace(bytes([13, 10]), bytes([10]))
    return b


def is_clean_vs_head(path: str) -> bool | None:
    """True if worktree content == content at the commit that last touched it."""
    lc = last_commit(path)
    if not lc["tracked"]:
        return None
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{lc['commit']}:{path}"],
                          capture_output=True)
    if blob.returncode != 0:
        return None
    disk = (ROOT / path).read_bytes()
    return (hashlib.sha256(_norm(blob.stdout, path)).hexdigest()
            == hashlib.sha256(_norm(disk, path)).hexdigest())


# ---------------------------------------------------------------- artifact set
GLOBS = [
    "experiments/csgqr/results/**/*.json",
    "experiments/csgqr/results/**/*.log",
    "experiments/csgqr/src/*.py",
    "experiments/csgqr/*.md",
    "experiments/sgqr_v2/results/**/*.json",
    "experiments/sgqr_v2/results/**/*.log",
    "experiments/sgqr_v2/src/*.py",
    "experiments/sgqr_v2/*.md",
    "artifacts/csgqr_probes/*",
]

# which script produced which artifact family
PRODUCER = {
    "forced_heldout_": "experiments/csgqr/src/eval_forced.py",
    "forced_L20_": "experiments/csgqr/src/eval_forced.py",
    "gpu_bench": "experiments/csgqr/src/bench_gpu.py",
    "layer_scan": "experiments/csgqr/src/measure_layers.py",
    "partial_gain_": "experiments/csgqr/src/calibrate_partial.py",
    "budget": "experiments/csgqr/src/budget.py",
    "arms_comparison": "experiments/csgqr/src/analyze_arms.py",
    "verify_csgqr": "experiments/csgqr/src/verify_csgqr.py",
    "verify_claims": "experiments/csgqr/src/verify_claims.py",
    "independent_audit": "experiments/csgqr/src/independent_audit.py",
    "confidence_analysis": "experiments/csgqr/src/analyze_confidence.py",
    "significance_analysis": "experiments/csgqr/src/analyze_significance.py",
    "threshold_sweep": "experiments/csgqr/src/sweep_threshold.py",
    "trace_manifest": "experiments/csgqr/src/trace_utility.py",
    "probe_analysis": "experiments/csgqr/src/probe_utility.py",
    "environment": "experiments/csgqr/src/envinfo.py",
    "eval_b": "experiments/csgqr/src/run_eval.py",
    "exp0_damage": "experiments/sgqr_v2/src/run_arms.py",
    "arms_ABCG": "experiments/sgqr_v2/src/run_arms.py",
    "recovery_H": "experiments/sgqr_v2/src/eval_recovery.py",
}


def producer_for(rel: str) -> str | None:
    name = Path(rel).name
    for key, script in PRODUCER.items():
        if name.startswith(key) or key in rel:
            return script
    return None


def main() -> int:
    order = commit_order()
    rows = []
    seen = set()
    for g in GLOBS:
        for p in sorted(ROOT.glob(g)):
            if not p.is_file() or "__pycache__" in str(p):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            lc = last_commit(rel)
            row = {
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
                "sha256_lf_normalized": hashlib.sha256(
                    _norm(p.read_bytes(), rel)).hexdigest(),
                "tracked": lc["tracked"],
                "carrying_commit": lc["commit"],
                "carrying_commit_date": lc["date"],
                "carrying_commit_subject": lc["subject"],
                "worktree_matches_commit": is_clean_vs_head(rel) if lc["tracked"] else None,
            }
            if rel.endswith(".json") or rel.endswith(".log"):
                prod = producer_for(rel)
                row["producer_script"] = prod
                if prod and (ROOT / prod).exists():
                    pc = last_commit(prod)
                    row["producer_last_commit"] = pc["commit"]
                    row["producer_last_commit_subject"] = pc["subject"]
                    if lc["commit"] and pc["commit"] and lc["commit"] in order and pc["commit"] in order:
                        row["producer_changed_after_result"] = order[pc["commit"]] > order[lc["commit"]]
                    else:
                        row["producer_changed_after_result"] = None
            rows.append(row)

    manifest = {
        "audit_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "audit_head": git("rev-parse", "HEAD"),
        "n_artifacts": len(rows),
        "untracked": [r["path"] for r in rows if not r["tracked"]],
        "worktree_diverged": [r["path"] for r in rows if r["worktree_matches_commit"] is False],
        "produced_by_since_changed_code": [
            {"path": r["path"], "producer": r.get("producer_script"),
             "producer_commit": r.get("producer_last_commit_subject")}
            for r in rows if r.get("producer_changed_after_result")
        ],
        "artifacts": rows,
    }
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"artifacts hashed        : {len(rows)}")
    print(f"untracked               : {len(manifest['untracked'])}")
    for u in manifest["untracked"]:
        print("   UNTRACKED", u)
    print(f"worktree != commit      : {len(manifest['worktree_diverged'])}")
    for u in manifest["worktree_diverged"]:
        print("   DIVERGED ", u)
    print(f"result predates producer: {len(manifest['produced_by_since_changed_code'])}")
    for u in manifest["produced_by_since_changed_code"]:
        print("   STALE-CODE", u["path"], "<-", u["producer"], "|", u["producer_commit"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
