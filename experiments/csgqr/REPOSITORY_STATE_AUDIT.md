# CSGQR — Repository State Audit

Written **before** any implementation file was modified in this session, as the
campaign brief requires. Every fact below is a recorded command output, not a
recollection. Commands were run from `C:/Users/clclc/RineyaExperiment` on
2026-08-24.

---

## 1. Headline finding

> **The remote branch `origin/research/csgqr-quantized-remnant` contains no CSGQR
> work. It is byte-identical to its base branch `origin/r2-healing-harness`.
> All CSGQR implementation, results and documentation exist only in this local
> working copy, in 15 unpushed commits plus uncommitted files.**

This is the exact failure mode the brief warned about, in its second form: not
"the branch is empty and nothing was ever built", but "substantive work exists
and was never pushed, so the branch a reviewer would clone is empty". Branch
creation is not implementation; a local commit is not a remote push.

---

## 2. Checked-out state

| Item | Value |
| --- | --- |
| Repository root | `C:/Users/clclc/RineyaExperiment` |
| Checked-out branch | `research/csgqr-quantized-remnant` |
| Local `HEAD` | `fb790a63c34d5f50fa19bbf857fa853b7c09877a` |
| Tracking status | `ahead 15` of `origin/research/csgqr-quantized-remnant` |
| Remote | `origin` -> `https://github.com/LUNCY579/RineyaBeta.git` (fetch + push) |

```
$ git status --short --branch
## research/csgqr-quantized-remnant...origin/research/csgqr-quantized-remnant [ahead 15]
```

---

## 3. Ref comparison

| Ref | SHA |
| --- | --- |
| local `HEAD` | `fb790a63c34d5f50fa19bbf857fa853b7c09877a` |
| `origin/research/csgqr-quantized-remnant` | `1664f54de0bbdfde45e311b75bbf3038dbfffe42` |
| `origin/r2-healing-harness` (base) | `1664f54de0bbdfde45e311b75bbf3038dbfffe42` |
| `origin/main` | `9e00be74098878290dc2b0d54cbffee4652c6088` |

The remote CSGQR branch and the base branch are **the same commit**.

```
$ git rev-list --left-right --count \
    origin/r2-healing-harness...origin/research/csgqr-quantized-remnant
0	0

$ git diff --stat origin/r2-healing-harness...origin/research/csgqr-quantized-remnant
(empty output)

$ git diff --name-status origin/r2-healing-harness...origin/research/csgqr-quantized-remnant
(empty output)
```

Against the local tip the picture is entirely different:

```
$ git rev-list --left-right --count origin/r2-healing-harness...HEAD
0	15

$ git diff --shortstat origin/r2-healing-harness...HEAD
 54 files changed, 14283 insertions(+), 143 deletions(-)

$ git diff --shortstat origin/r2-healing-harness...HEAD -- experiments/csgqr
 28 files changed, 9560 insertions(+)
```

`ls-remote` confirms the remote label points at the base commit and that the
branch has never received a CSGQR push:

```
$ git ls-remote --heads origin
d5bc8ba7d7784b0feef672a2ab942a48412a5349	refs/heads/exp/stabilizer-matrix
9e00be74098878290dc2b0d54cbffee4652c6088	refs/heads/main
1664f54de0bbdfde45e311b75bbf3038dbfffe42	refs/heads/r2-healing-harness
1664f54de0bbdfde45e311b75bbf3038dbfffe42	refs/heads/research/csgqr-quantized-remnant
1556b4c74bacb30602290a71a30a7971ba355db5	refs/heads/research/pre-deletion-recovery-routing-v0.3
```

Note: `git fetch --prune` during this audit deleted two stale remote-tracking
refs (`research/pre-deletion-recovery-routing-v0.1` and `-v0.2`) and advanced
`-v0.3` from `1664f54` to `1556b4c`. Those are other branches owned elsewhere;
nothing was pushed to them and nothing about them was changed locally beyond the
remote-tracking refs that `--prune` maintains.

---

## 4. The 15 local commits

Four are CSGQR work; eleven arrived with the `origin/main` merge.

| SHA | Date (local) | Subject |
| --- | --- | --- |
| `fb790a6` | 2026-08-23 21:36 | Measure the mechanism: significance works, confidence does not, and the state does |
| `8627c71` | 2026-08-23 21:05 | Add the measurement and analysis harness for the CSGQR campaign |
| `272e247` | 2026-08-23 20:53 | Implement the CSGQR quantized remnant, and fix two bugs it surfaced |
| `3c051f1` | 2026-08-23 20:28 | Bring the CSGQR branch up to main *(merge commit)* |
| `9e00be7` ... `b999cb0` | 2026-08-21 | eleven commits inherited from `origin/main` via the merge |

```
$ git log --all --oneline --grep='CSGQR'
8627c71 Add the measurement and analysis harness for the CSGQR campaign
272e247 Implement the CSGQR quantized remnant, and fix two bugs it surfaced
3c051f1 Bring the CSGQR branch up to main
```

No commit anywhere in the repository matches `--grep='confidence-gated'`.

---

## 5. Base branch contains zero CSGQR content

```
$ git ls-tree -r --name-only origin/r2-healing-harness | grep -i -E 'csgqr|remnant'
(no matches)

$ git grep -l -i "remnant" origin/r2-healing-harness -- .
(no matches)
```

So the previously-recorded claim that this branch began with no CSGQR
implementation is confirmed independently here, at the level of the remote base
tree rather than from the prior session's notes.

---

## 6. Working tree: what is committed vs. not

Tracked-and-committed under `experiments/csgqr` (28 files, all added by the three
CSGQR commits): `README.md`, 18 modules under `src/`, and result files
(`budget.json`, `environment.json`, `layer_scan.json`, `verify_csgqr.json`, and
four JSON files under `results/trace_L20_int4/`).

**Uncommitted, and therefore at risk:**

| Path | State | Nature |
| --- | --- | --- |
| `experiments/csgqr/src/run_eval.py` | modified | ranking/group-count guard (8 added lines) |
| `experiments/csgqr/EXPERIMENT_LOG.md` | untracked | 335 lines |
| `experiments/csgqr/METHOD_AUDIT.md` | untracked | 291 lines |
| `experiments/csgqr/NEGATIVE_RESULTS.md` | untracked | 205 lines |
| `experiments/csgqr/run_tier1.sh` | untracked | 57 lines |
| `experiments/csgqr/src/analyze_arms.py` | untracked | 272 lines |
| `experiments/csgqr/src/verify_claims.py` | untracked | 399 lines |
| `experiments/csgqr/results/eval_*/` x 10 | untracked | the entire Tier-1 arm sweep |
| `experiments/csgqr/results/trace_L20_int4/threshold_sweep_max_prob.json` | untracked | frontier sweep |

Ten completed evaluation runs — roughly 70 minutes of GPU time and the campaign's
only generation-level evidence — were untracked at the start of this session.

`.gitignore` deliberately excludes the bulky per-case artifacts
(`experiments/csgqr/results/**/results.jsonl`, `**/traces.jsonl`,
`experiments/csgqr/results/*.log`), so `summary.json`, `per_category.json` and
`run_manifest.json` are the committable evidence for each arm. That is a
defensible policy, but it means **`verify_claims.py` must be able to verify every
committed claim from the committed summaries alone**, which is checked in Stage II
rather than assumed.

---

## 7. Documentation files the brief requires: present vs. absent

Present: `README.md`, `METHOD_AUDIT.md`, `EXPERIMENT_LOG.md`,
`NEGATIVE_RESULTS.md` (the last three untracked).

**Absent** — and several are referenced by name from files that do exist, so the
existing documentation contains dangling cross-references:
`REPOSITORY_STATE_AUDIT.md` (this file, now created), `METHOD_SPECIFICATION.md`,
`SPEC_TO_CODE_MATRIX.md`, `BUG_LEDGER.md`, `METHOD_AND_CODE_AUDIT.md`,
`IMPLEMENTATION_READINESS.md`, `EXPERIMENT_PLAN.md`, `FINDINGS.md`,
`GPU_BENCHMARKS.md`, `CALIBRATION_AND_ROUTING.md`,
`CONFIDENTLY_WRONG_ANALYSIS.md`, `STATISTICAL_ANALYSIS.md`,
`NOVELTY_AND_RELATED_WORK.md`, `LIMITATIONS.md`,
`MAXIMUM_POTENTIAL_ROADMAP.md`, `RESEARCH_VERDICT.md`.

`EXPERIMENT_LOG.md` section 12 points at `FINDINGS.md` for the Tier-1 result;
`METHOD_AUDIT.md` section 6 points at `RESEARCH_VERDICT.md`. Neither exists. The
Tier-1 sweep therefore has **no committed interpretation** anywhere.

Also absent, and named in the brief: `scripts/verify/verify_csgqr.py` and
`scripts/verify/verify_csgqr_claims.py`. The equivalents live at
`experiments/csgqr/src/verify_csgqr.py` and
`experiments/csgqr/src/verify_claims.py`. That is a path discrepancy, not a
missing capability.

---

## 8. Worktrees

```
$ git worktree list
C:/Users/clclc/RineyaExperiment             fb790a6 [research/csgqr-quantized-remnant]
C:/Users/clclc/RineyaExperiment-stabilizer  d5bc8ba [exp/stabilizer-matrix]
```

The second worktree holds unrelated stabilizer-matrix work at
`d5bc8ba7d7784b0feef672a2ab942a48412a5349`. It contains no CSGQR files and is
out of scope; it will not be touched.

---

## 9. Stash, reflog, dangling objects

`git stash list` is empty. The reflog shows the branch was created by checkout
from `exp/stabilizer-matrix` at `1664f54`, then the `origin/main` merge, then the
three CSGQR commits — no rewrites, no force operations.

Two dangling commits exist. Both were inspected and belong to the earlier
KV-compression programme, not to CSGQR:

- `b43d3e8` "Design the KV-compression long-context study, with feasibility tooling"
- `3fcfcfd` "Correct a power claim I had wrong, and make identity counts survive a clone"

No CSGQR work is hiding in unreachable objects.

---

## 10. Filesystem search

```
$ find . -not -path './.git/*' -not -path './.venv/*' \
       \( -iname '*csgqr*' -o -iname '*quantized*remnant*' \) -print
./experiments/csgqr
./experiments/csgqr/results/verify_csgqr.json
./experiments/csgqr/src/verify_csgqr.py
./experiments/csgqr/src/__pycache__/verify_csgqr.cpython-311.pyc
```

Nothing outside `experiments/csgqr`. No second copy, no abandoned variant.

---

## 11. Environment recorded at audit time

From `results/environment.json`, cross-checked against `nvidia-smi`:

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop, cc 8.9, 36 SMs, 8187.5 MiB total |
| Driver | 610.62 |
| torch | 2.13.0+cu130 (venv); the **ambient** interpreter is CPU-only |
| transformers | 5.15.0 |
| Python | 3.11.15 (`.venv/Scripts/python.exe`) |
| OS | Windows 11 build 26200 |
| Checkpoint | `artifacts/rineya-core-en32k-28L` — Qwen3, 28L, hidden 1024, d_ff 3072, vocab 32768, tied embeddings, bf16 |
| Weights | 948,078,864 B, sha256 `a6bc04ca357d446d...` |

The recorded `environment.json` was written under SHA `3c051f1`, two commits
behind the current tip, so it predates the harness and analysis modules. It will
be regenerated.

---

## 12. State of evidence at audit time

The Tier-1 arm sweep had completed all nine arms (uncommitted). Mean quality
score on `chat_dev`, 510 cases, layer 20:

| arm | mean_quality_score |
| --- | ---: |
| b0-dense | 0.833596 |
| b1-delete | 0.822243 |
| b2-fp-all | 0.834441 |
| b3-int8-all | 0.830255 |
| b4-int4-all | 0.835583 |
| b5-random-25 | 0.828188 |
| b6-static-25 | 0.819469 |
| b7-conf-whole | 0.825198 |
| b10-csgqr | 0.824382 |
| b13-wrong-layer | 0.825704 |

**Two of these are anomalous on their face and are recorded here as open
anomalies, not as results:**

1. **Precision is non-monotone.** INT4-all (0.835583) scores above INT8-all
   (0.830255) *and* above dense (0.833596). The gap between the two quantized
   arms is 0.005328, which is 6.3x the |b0 - b2| noise floor of 0.000845, and it
   runs the wrong way.
2. **Static top-significance at rho=0.25 (0.819469) scores below exact deletion
   (0.822243), and below random groups at the same budget (0.828188).** Running
   the groups the significance analysis ranked most useful is worse than running
   none, and worse than running an arbitrary quarter.

Both must be explained before any arm number in this table is treated as
evidence about CSGQR. Neither is used as evidence in this audit.

---

## 13. Audit conclusions

1. The remote branch is empty of CSGQR work and identical to its base. **Pushing
   is a completion requirement, not a formality.**
2. Substantial local implementation exists (about 6,400 lines under `src/`, 9,560
   insertions under `experiments/csgqr`) and is of a serious standard: the
   packed-INT4 layout, the group-additivity argument, the hook location and the
   analytic budget were all checked rather than asserted.
3. Ten evaluation runs, three of the four existing audit documents, and two
   analysis modules were **uncommitted** and would have been lost by any
   clean-tree operation.
4. Sixteen of the twenty required documentation files are missing, including
   every Stage II gate document and the final verdict, and existing files
   cross-reference several of them.
5. Two Tier-1 results are unexplained. Under this repository's standing rule —
   an unexplained result is indistinguishable from a bug — they are treated as
   defects to be chased, not as findings.
6. No unrelated branch, worktree, or sealed dataset was modified in the course of
   this audit. The audit was read-only apart from creating this file.

**Immediate actions, in order:** commit the uncommitted evidence and push the
branch so that the work exists remotely; then proceed to Stage I with the
line-by-line code review, the spec-to-code matrix and the bug ledger.
