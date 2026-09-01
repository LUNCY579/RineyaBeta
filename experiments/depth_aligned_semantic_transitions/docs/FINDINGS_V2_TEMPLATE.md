# DAST V2 — optimization conflict — findings (TEMPLATE)

**Status: UNRUN.** Fill this document only after `PREREGISTRATION_V2.md` is signed
and the nine `v2_s{17,29,43}_{ordered,reverse,final_repeated}` runs complete.
Until then every table cell is empty on purpose.

**Hypothesis under test.** Forward state supervision produces greater optimization
conflict with the language-model objective than matched reverse or flat-target
supervision.

**Explicitly not under test.** Whether ordered DAST improves held-out reasoning.
That V1 hypothesis is closed. Accuracy below is a consistency check, not a gate.

**Sealed evaluation:** CLOSED. Do not open TEST to resolve an ambiguous DEV result.

**Preregistration:** `PREREGISTRATION_V2.md` — DRAFT until the freeze block is
filled. Record the frozen SHA-256 here before interpreting any number.

```text
preregistration_sha256: PENDING
commit: PENDING
```

## 1. Final teacher-forced `L_state` (V2-G1)

| arm | s17 | s29 | s43 | mean | n |
|---|---:|---:|---:|---:|---:|
| `state_ordered` |  |  |  |  |  |
| `state_reverse` |  |  |  |  |  |
| `state_final_repeated` |  |  |  |  |  |

| contrast | predicted | observed mean diff | same sign in all seeds? | margin | gate |
|---|---|---:|---|---:|---|
| C1 ordered − reverse | ordered higher by ≥ 0.005 |  |  | 0.005 | V2-G1 |
| C2 ordered − final_repeated | ordered higher by ≥ 0.05 |  |  | 0.05 | V2-G1 |

**V2-G1:** PASS / FAIL

## 2. Training `loss/lm` path (V2-G2)

Trapezoidal AUC over `optimizer_step` from `metrics.jsonl`, plus the final logged
`loss/lm`.

| arm | s17 AUC | s29 AUC | s43 AUC | mean AUC | mean final `loss/lm` |
|---|---:|---:|---:|---:|---:|
| `state_ordered` |  |  |  |  |  |
| `state_reverse` |  |  |  |  |  |
| `state_final_repeated` |  |  |  |  |  |

| contrast | predicted | observed | same sign in all seeds? | gate |
|---|---|---|---|---|
| C3 ordered − reverse AUC | ordered higher |  |  | V2-G2 |

**V2-G2:** PASS / FAIL / NOT EVALUATED (if V2-G1 failed)

## 3. Consistency check — strict generated-answer accuracy

Not a gate. Expectation from V1: ordered ≤ reverse and ordered ≤ final-repeated.
A reversal does not pass V2. A replication of the V1 deficit does not reopen V1.

| arm | s17 | s29 | s43 | mean |
|---|---:|---:|---:|---:|
| `state_ordered` |  |  |  |  |
| `state_reverse` |  |  |  |  |
| `state_final_repeated` |  |  |  |  |

## 4. Verdict

Write only after the gates above are filled. Allowed conclusions are those in
`PREREGISTRATION_V2.md` §9. Forbidden: reopening V1, starting operators, opening
TEST, treating mechanism decoding scores as evidence that training built a
depth trajectory.

## 5. Reproduce

```bash
python scripts/summarize_training_dynamics.py --prefix v2_
python scripts/summarize_runs.py --prefix v2_
```

## 6. Claim boundary

Three seeds. Unsigned-until-frozen V2 preregistration. TEST sealed. V1 remains
CLOSED — NEGATIVE. Nothing here is a Rineya capability claim.
