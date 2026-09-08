# HLGA-1 diagnostic debugging guide

This guide is an engineering aid for the existing bounded open DEV stage. It does not open P3, authorize training, change prompts, change registered endpoints, or convert exploratory results into confirmation.

## Why the reporting layer was hardened

The original `family|length|depth` table has one case per exact cell. Those cells are useful for locating a failure, but an `n=1` cell is a poor basis for deciding whether a failure is systematic. Diagnostic schema v2 therefore retains every original exact cell and adds margins by family, length, depth, and answerability. It also duplicates immutable evidence/question position metadata into each arm row, so an individual arm artifact can be debugged without joining against `open_dev_cases.json`.

No prompt token IDs, answers, case order, parser rule, strict/trimmed exactness definition, arm order, decoding cap, primary contrast, topology, or gate definition changes in this hardening increment.

## Triage order after a future P3 run

1. Check the P3 receipt and workload/cache guards first. A failed guard is an execution failure, not a model-quality result.
2. Check A0 global readout health. If A0 is broadly empty, repetitive, budget-exhausted, or wrong, treatment ties are inconclusive rather than evidence of compatibility.
3. Check the three identity pairs A0/A4, A2/A5, A3/A6. Any token mismatch invalidates interpretation before training.
4. Read the registered A3-versus-A0 contrast, then use margins only to localize the pattern: family isolates task type, length isolates window-boundary sensitivity, depth isolates evidence placement, and answerability separates retrieval failure from abstention behavior.
5. Use `termination`, `generated_tokens`, `unique_generated_tokens`, and repeat statistics to distinguish answer errors from decoding pathologies. These are diagnostic-only fields.
6. Use A2-versus-A3 to ask whether prefix sinks help at equal local budget; A8-versus-A2 to ask whether periodic globals matter; and A9-versus-A2 to ask whether global placement matters at equal modeled work. Do not reinterpret A8 as compute-matched.

## Failure signatures

- **Native/global or global/full-stream P2 mismatch:** stop. Debug numerical/backend/cache mechanics; P3 stays closed.
- **Only local/full-stream mismatch:** stop. Debug eviction, absolute positions, block boundaries, sink overlap, and compact-cache accounting.
- **A0 weak across nearly everything:** endpoint/readout is likely underpowered for the checkpoint; do not call an A3 tie noninferiority.
- **A3 degrades mainly near N=513/1025:** inspect boundary/eviction behavior before invoking a general reasoning explanation.
- **A2 weak but A3 recovers:** evidence is consistent with prefix sinks helping this open diagnostic, not with general long-context improvement.
- **A8 weak while A2 is healthier:** evidence is consistent with periodic global layers mattering in this diagnostic.
- **A9 differs from A2:** placement matters even at the registered equal modeled work/cache declaration; the simple periodic rule should not be treated as uniquely justified.
- **High budget exhaustion/repetition:** classify as decoding/readout pathology before interpreting semantic accuracy.

## What still requires separate evidence

P3 remains a 511-1025 token boundary/readout diagnostic. It cannot establish 4K/8K/16K reasoning, whole-process memory savings, latency, throughput, training benefit, family-wide compatibility, or promotion. P4 gate training and P5 confirmation remain separately gated. Any new precision, kernel, checkpoint, context regime, data source, or optimizer registration requires a new bound receipt rather than inheriting this one.
