# Guarded branch retirement — 2026-09-09

User authorized applying the recommended RineyaBeta archives/retirements.
The initial remote snapshot has eight live branches and eleven existing retired tags.
No open pull requests were present; the write job rechecks this before acting.

This is an isolated temporary maintenance branch descending from the composition
CI-only fix candidate. Main, Cornerstone2, old evidence files, pretrained assets,
experiment algorithms and scientific gates are not edited by this maintenance.
The only proposed experiment-branch file change is pytest importlib discovery in
`.github/workflows/official-family.yml` (candidate 5785ef1), resolving the duplicate
`test_native.py` module-name collision. It does not exclude tests or relax checks.

## Conditional operations

1. Query: test the exact official-family head 5500e87, verify ancestry from
   e579450, archive `research/query-refinement-attention-v1` at e579450, then remove
   its live branch name. The official-family query branch does not move.
2. Composition: test candidate 5785ef1 with the shared plus inherited suites;
   if passing, fast-forward `research/composition-qrr-official-family-v1` from
   ffb0e4b to 5785ef1 and retire `research/composition-qrr-v1` at 690dcf2.
3. Hybrid: run the inherited `verify --require-hf` command on 2b1131e under its
   pinned CPU runtime; if passing, preserve a196318 and the continuation with
   separate annotated tags, then fast-forward the canonical
   `research/hybrid-local-global-attention-v1` branch to 2b1131e and remove the
   redundant `agent/hlga-continuation-20260908` branch name.

A failed or missing experiment gate retains that pair of live branches. Archival
is not approval of historical P2 amendments, post-hoc numerical rationales, or
model-quality claims. In particular, the hybrid continuation's protocol changes
and diagnostic reports are preserved as their own research history; this action
does not repeat or relicense its real-model campaign. No trained weights are
loaded and no optimizer is run. P4/P5/training remain closed.

## Transaction safeguards

The script validates exact initial branch SHAs, source-bound test receipts,
ancestry, the narrow composition diff and the hybrid no-training/no-claim flags.
It does not touch unrelated refs. Recovery tags are created and verified on the
remote before any branch deletion. One atomic push applies exact-SHA leases to
all retired/advanced refs; concurrent work rejects the transaction instead of
being overwritten. Existing tags are never overwritten. The temporary maintenance
branch is itself tagged and removed in that transaction to avoid permanent clutter.

The actual report is retained as a workflow artifact and in the annotated tag
`retired/audit/branch-retirement-20260909`, with the original and resulting remote
snapshots, archive targets, test outcomes and workflow provenance. The maintenance
source remains available at `retired/maintenance/branch-retirement-20260909`.

Local maintenance-script tests: 6 passed, including a real local bare-Git
simulation proving stale deletion leases reject the whole atomic transaction and
preserve competing commits. The same tests run in cloud CI before any writes.
These tests do not establish that the three experiment gates will pass.

Restore an archived branch with a new branch pointing at its annotated tag's
peeled commit. Never move an existing branch or overwrite an archive to restore it.
This workflow is single-use and pinned to this reviewed snapshot; it deliberately
refuses reruns against changed refs or preexisting archive names.
