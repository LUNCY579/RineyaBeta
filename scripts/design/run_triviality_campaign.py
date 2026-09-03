#!/usr/bin/env python3
"""Drive the six-arm triviality control preregistered in
docs/PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md.

ONE SOURCE OF TRUTH FOR THE MATCHING. Every arm is generated from the table
below, so "matched on budget / sinks / recent-frac / seed / model" is a property
of the code rather than of six command lines that have to agree. The analyser
still diffs the manifests afterwards (prereg 6.2), because a shared table
guarantees the flags agree and not that the harness honoured them.

ONE GPU JOB AT A TIME. This host is a single RTX 4070 8 GB shared with the
desktop and with other worktrees; a window can close mid-run, and when it did the
pass did not crash, it ran ~100x slower and wrote a partial file. So: the GPU is
checked before every arm, each arm is a separate process writing its own
directory, and a campaign killed half way leaves completed arms intact and no
partial ones (prereg 8).

Usage:
    python scripts/design/run_triviality_campaign.py --gate         # 10-case repro gate
    python scripts/design/run_triviality_campaign.py --dataset rag_long4k
    python scripts/design/run_triviality_campaign.py --all
    python scripts/design/run_triviality_campaign.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/kv_significance/src/run_eval_sig.py"
RESULTS = ROOT / "experiments/kv_significance/results"

MODEL = "artifacts/rineya-core-en32k-28L"
SEED = 20260826
KEEP_FRAC = 0.25
SINKS = 4
RECENT_FRAC = 0.15
WINDOW = 16
CAP_MIB = 7000          # honest OOM inside our own budget beats taking the card down

# (arm id, tag stem, select, pool, pool_back, pool_fwd)
# Order is the RUN order and it is deliberate. A campaign killed after three arms
# still yields the primary contrast: A is the reference every comparison needs, G is
# the decisive arm (the strongest trivial selector), and N is the contiguity-matched
# floor without which G's number cannot be interpreted. B and C -- the ablations of
# the mechanism's own contribution -- come next, and E last, because E is the weakest
# member of a maximum and cannot change it.
#
# Arm F is NOT here. long{4k,16k}-random-k0.25 are already committed at exactly this
# configuration and are reused (prereg 2.4), conditional on the reproduction gate.
ARMS = [
    ("A", "c4-sig-b6f0",         "significance",     7, 6, 0),
    ("G", "c4-lex-docbm25-b6f0", "lexical-doc-bm25", 7, 6, 0),
    ("N", "c4-random-prepool",   "random-prepool",   7, 6, 0),
    ("Ndoc", "c4-random-doc",    "random-doc",       7, 6, 0),
    ("D", "c4-lex-idf-b6f0",     "lexical-idf",      7, 6, 0),
    ("B", "c4-sig-centred",      "significance",     7, 3, 3),
    ("C", "c4-sig-nopool",       "significance",     1, None, None),
    ("E", "c4-lex-naive-b6f0",   "lexical-naive",    7, 6, 0),
]

DATASETS = [("rag_long4k", 200), ("rag_long16k", 100)]

# The committed arm the 10-case reproduction gate is checked against (prereg 6.1).
GATE_REFERENCE = "long4k-sig-k0.25-rf0.15__rag_long4k"


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(limit_mib: int, wait: bool) -> None:
    """Refuse to start while another job holds the card."""
    while True:
        used = gpu_used_mib()
        if used is None:
            print("  nvidia-smi unavailable -- proceeding without the guard")
            return
        if used <= limit_mib:
            print(f"  gpu clear: {used} MiB used (limit {limit_mib})")
            return
        msg = (f"  gpu BUSY: {used} MiB used, limit {limit_mib}. Another worktree is "
               f"running a job.")
        if not wait:
            raise SystemExit(msg + " Re-run with --wait, or wait for it to finish.")
        print(msg + " waiting 120 s...", flush=True)
        time.sleep(120)


def cmd_for(arm, dataset: str, limit: int, tag: str, cap: int) -> list[str]:
    _id, _stem, select, pool, back, fwd = arm
    c = [sys.executable, str(RUNNER),
         "--dataset", dataset,
         "--model", MODEL,
         "--keep-frac", str(KEEP_FRAC),
         "--sinks", str(SINKS),
         "--recent-frac", str(RECENT_FRAC),
         "--window", str(WINDOW),
         "--pool", str(pool),
         "--select", select,
         "--seed", str(SEED),
         "--limit", str(limit),
         "--cap-mib", str(cap),
         "--tag", tag]
    if pool > 1:
        # BOTH or NEITHER -- Policy raises otherwise, and the reason it raises is
        # that --pool 7 --pool-back 6 alone silently yields a window that reaches
        # three positions FORWARD.
        c += ["--pool-back", str(back), "--pool-fwd", str(fwd)]
    return c


def complete(out_dir: Path, expect: int) -> bool:
    man, res = out_dir / "run_manifest.json", out_dir / "results.jsonl"
    if not (man.exists() and res.exists()):
        return False
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
    except Exception:
        return False
    n = sum(1 for l in res.read_text(encoding="utf-8").splitlines() if l.strip())
    return m.get("case_count") == expect and n == expect


def run_arm(arm, dataset: str, limit: int, tag: str, args) -> bool:
    out_dir = RESULTS / f"{tag}__{dataset}"
    if complete(out_dir, limit) and not args.force:
        print(f"[{arm[0]}] {tag}__{dataset}: already complete, skipping")
        return True
    cmd = cmd_for(arm, dataset, limit, tag, args.cap_mib)
    print(f"\n[{arm[0]}] {tag}__{dataset}\n  {' '.join(cmd[1:])}", flush=True)
    if args.dry_run:
        return True
    # WAIT FOR THE CARD **BEFORE** TOUCHING THE OUTPUT DIRECTORY. The other order
    # empties a complete 200-case arm and then raises SystemExit because the GPU is
    # busy -- destroying data and running nothing. Reachable with --force on this
    # host's shared card, which is precisely when --force gets used.
    wait_for_gpu(args.min_free_check, args.wait)
    if out_dir.exists():
        # A partial artifact is deleted, not analysed. An incomplete arm that
        # survives to the analyser is how a null gets manufactured from missing data.
        stale = sum(1 for _ in out_dir.glob("*"))
        print(f"[{arm[0]}] removing {stale} stale file(s) in {out_dir.name}")
        for f in out_dir.glob("*"):
            f.unlink()
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT))
    ok = p.returncode == 0 and complete(out_dir, limit)
    print(f"[{arm[0]}] {'OK' if ok else 'FAILED'}  {time.time() - t0:.0f}s", flush=True)
    if not ok and out_dir.exists() and not args.keep_partial:
        print(f"[{arm[0]}] deleting partial artifacts (prereg 8)")
        for f in out_dir.glob("*"):
            f.unlink()
    return ok


def gate(args) -> bool:
    """Prereg 6.1 -- 10 cases of arm A against the committed arm, byte for byte."""
    tag = "c4-gate-sig-b6f0"
    arm = ARMS[0]
    if not run_arm(arm, "rag_long4k", 10, tag, args):
        print("GATE: the gate run itself failed")
        return False
    if args.dry_run:
        return True
    new = [json.loads(l) for l in
           (RESULTS / f"{tag}__rag_long4k/results.jsonl").read_text(
               encoding="utf-8").splitlines() if l.strip()]
    ref_path = RESULTS / GATE_REFERENCE / "results.jsonl"
    if not ref_path.exists():
        print(f"GATE: no committed reference at {ref_path}")
        return False
    ref = {}
    for l in ref_path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ref[r["id"]] = r
    same = sum(1 for r in new if r["id"] in ref and r["reply"] == ref[r["id"]]["reply"])
    print(f"\nGATE: {same}/{len(new)} replies byte-identical to {GATE_REFERENCE}")
    if same < len(new):
        for r in new:
            if r["id"] in ref and r["reply"] != ref[r["id"]]["reply"]:
                print(f"  DIFF {r['id']}\n    committed: {ref[r['id']]['reply']!r}"
                      f"\n    now      : {r['reply']!r}")
    print("GATE: " + ("PASS -- the campaign inherits the committed arms' provenance"
                      if same == len(new) else
                      "FAIL -- arms stay internally comparable, but NO result may be "
                      "compared to a committed number from the earlier code state. "
                      "Report this as a finding (prereg 6.1)."))
    # Return the GATE's outcome, not merely "the run completed". Returning True on a
    # failed gate made `--gate` exit 0 while printing FAIL, so a scripted caller read
    # a failed provenance check as success.
    return same == len(new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=[d for d, _ in DATASETS])
    ap.add_argument("--arm", action="append",
                    help="run only these arm ids (A..F); repeatable")
    ap.add_argument("--all", action="store_true", help="both datasets, all arms")
    ap.add_argument("--gate", action="store_true", help="only the 10-case repro gate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run completed arms")
    ap.add_argument("--keep-partial", action="store_true",
                    help="do NOT delete a failed arm's artifacts (debugging only)")
    ap.add_argument("--wait", action="store_true",
                    help="poll until the GPU frees instead of refusing to start")
    ap.add_argument("--cap-mib", type=int, default=CAP_MIB,
                    help="hard ceiling on THIS process's GPU allocation. Lower it "
                         "when another project holds part of the card: the cap keeps "
                         "the failure inside our own budget instead of taking the "
                         "device down under someone else's job. Recorded in every "
                         "manifest, so all arms in a cell must share it.")
    ap.add_argument("--min-free-check", type=int, default=3000,
                    help="refuse to start while more than this many MiB are in use")
    args = ap.parse_args()

    if args.gate:
        raise SystemExit(0 if gate(args) else 1)

    todo = DATASETS if (args.all or not args.dataset) else \
        [(d, n) for d, n in DATASETS if d == args.dataset]
    arms = [a for a in ARMS if not args.arm or a[0] in args.arm]

    print(f"campaign: {len(arms)} arm(s) x {len(todo)} dataset(s) "
          f"= {len(arms) * len(todo)} run(s)")
    done, failed = [], []
    for dataset, limit in todo:
        for arm in arms:
            (done if run_arm(arm, dataset, limit, arm[1], args) else failed).append(
                f"{arm[0]}/{dataset}")
    print(f"\ncomplete: {len(done)}  {done}")
    if failed:
        print(f"FAILED:   {len(failed)}  {failed}")
        print("A partial campaign yields a partial report. Do not read a missing "
              "arm as a null (prereg 8).")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
