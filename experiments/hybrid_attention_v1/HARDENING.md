# HLGA-1 execution and comparison hardening

Reviewed parent: `f36425368b098d41fc4802503ba1f43f372050da`.
This increment is infrastructure, not a pretrained Rineya result. The canonical
`protocol.json` and `COMPATIBILITY_CONTRACT.md` remain in force. Training P4,
independent confirmation P5, historical sealed suites and performance promotion
remain closed. No checkpoint, tokenizer, legacy result, shared trainer, other
branch, tag or main is modified by this increment.

## Reconciliation with the preserved package

The earlier downloadable package targeted `6770e67...` and eight arms. The live
branch subsequently introduced `study.py`, ten arms, an A3-vs-A0 primary contrast,
A8 all-local and A9 frontloaded-placement controls, and runtime-bound accounting.
Do not apply the obsolete package or publishing helper over that newer work.

The useful missing component is now adapted as `matching.py`: a strict comparison
**declaration** auditor. The live `study.py` remains the only inference cost
planner and the source of frozen contrast names/directions. No duplicate geometry
class, second attention-FLOP oracle, stale eight-arm protocol or old unit receipt
is promoted as current evidence. `matching plan` delegates to `study.plan()`.

## Bugs reproduced and corrected

| Finding | Correction | Evidence scope |
|---|---|---|
| Nonfinite logits became ordinary argmax token IDs rather than a failed generation | Check all streaming logits, including intermediate prefill chunks, before accepting output | Tiny random fixture regression tests, not a Rineya instability finding |
| Invalid streaming input or embedding/RoPE failure occurred before the cache-reset handler | Wrap the entire streaming forward; reset generation state before validating a new request | Request-isolation and negative-path fixtures |
| Finite, opposite-sign FP32 extrema overflowed the diagnostic subtraction, risking an unserializable failed receipt | Compute the error magnitude in FP64 and suppress nonfinite diagnostic magnitudes | Synthetic numerical regression |
| Markdown contract changes and source edits during execution were not bound to completion | Include experiment-root Markdown in `code_identity`; recheck source identity before success | Artifact/provenance fixtures |
| A receipt did not bind the JSON result files alongside it | Record JSON artifact membership, sizes and hashes; require the bound P2 artifacts before P3 | Accidental-edit detection, not authentication |
| Required native verification could accept skipped tests or an empty collection | Require nonempty successful tests and zero skips with `--require-hf` | Verifier negative-path tests |

Failure-state corrections do not change attention connectivity, gate formulas,
healthy finite logits, prompts, decoding policy, case order, arm membership or
registered output caps. Absolute evidence/question spans and their gap are added
as metadata only: `depth` remains the requested filler-insertion fraction, not an
absolute position or a measured retrieval distance. For absent cases the displayed
record is a distractor, not support. Both two-hop facts remain co-located.

Checking logits for finiteness may synchronize CUDA and adds diagnostic overhead.
This is a correctness runner, not a speed benchmark. Existing fixed-order/shared-
hardware timing remains non-causal. New measurements cannot be mixed with old
fingerprints, and SDPA shape counters are not measured physical GPU instructions.

## Comparison declarations

```bash
python -m experiments.hybrid_attention_v1.matching plan
python -m experiments.hybrid_attention_v1.matching audit-pair \
  --left LEFT_LEDGER.json --right RIGHT_LEDGER.json \
  --contrast sink_at_equal_budget --mode frozen_workload
```

The two input files are explicit JSON metadata, not datasets. Required identity
fields include checkpoint, code, tokenizer, evaluation manifest, prompt/decoding,
environment and measurement protocol, plus each source receipt reference. Declare
`target_config`, the complete executed `spec`, backend, dtype, quantization, batch,
prefill chunk/block sizes, case count, context length, output cap and padding/
packing status. `spec` is checked against `arm_spec`, so an all-global treatment
cannot silently be relabeled as sink-local attention. A gate identity comparison
requires an independently verified identity declaration.

A ledger covers a common diagnostic context-length cell; `evaluation_cases` is
the number of paired cases in that ledger, bounded above by the 72-case study.
Use the same evaluation-manifest identity and exact membership for both arms.
Do not pass a 4096-token ledger as the registered boundary study. Multiple cell
ledgers do not authorize extra cases, retuning, or extra evaluations.

For future, separately authorized gate training, declare the same training pool,
loss and loss-mask protocol, trainable allowlist, optimizer/schedule, selection
rule, teacher/accounting policy and registered seed. Token-matched runs require
64 updates, identical order and supervised tokens. The registered microbatch is
one, accumulation eight and sequence length 1024; input and supervised counts
must be consistent. This is future-registration validation, not a trainer.

Equal-compute comparisons are separate: A4's reference is 64 updates, candidates
are capped at 128, and full compute includes student forward/backward, optimizer
and teacher under a common accounting policy. Declare source profiler receipts
and the shared sampling stream, with realized steps/tokens/order recorded.
Attention-only or forward-only estimates cannot satisfy this check. Device-time
accounting additionally requires declared isolation and randomized ordering.
The prespecified symmetric mismatch `abs(a-b)/max(a,b)` cannot exceed 2%; this is
a compute-matching convention, not an accuracy margin or an equivalence test.

The auditor verifies declarations, **not their authenticity**. Review the actual
source receipts, lineage, profiler and hardware logs independently. Passing it
never sets a training, sealed-evaluation, speed or quality-improvement flag true.
It is intentionally not a new execution gate that overrides existing approvals.

## Evidence handling and next permitted work

The final `receipt.json` is authoritative for run disposition; intermediate arm
files and summaries are provisional if the receipt fails or is missing. Its
`artifacts` map binds the JSON outputs, excluding the receipt itself to avoid a
hash cycle. Modification, omission, addition or symlinking of bound artifacts
invalidates that bundle. Hashes detect accidental changes, not maliciously forged
measurements or authorization. Real model identity still requires reviewed weight
lineage: an arbitrary same-shaped checkpoint is not proven to be canonical R2.

Root source/contract changes invalidate old qualifications. Historical reports
under `reports/` are not in the code fingerprint, so evidence-only commits do not
invalidate the source they document. Never overwrite historical receipts.

```bash
python -m unittest experiments.hybrid_attention_v1.tests.test_matching \
  experiments.hybrid_attention_v1.tests.test_hardening -v
python -m experiments.hybrid_attention_v1.verify --run-id hardening-native-001 --require-hf
# Actual local R2 assets are required for these; do not substitute random weights.
python -m experiments.hybrid_attention_v1.run audit --run-id hardening-r2-audit-001
python -m experiments.hybrid_attention_v1.run qualify --run-id hardening-r2-qualify-001
python -m experiments.hybrid_attention_v1.run diagnose --run-id hardening-r2-dev-001 \
  --receipt experiments/hybrid_attention_v1/runs/hardening-r2-qualify-001/receipt.json
```

Use matching device/backend/block settings across P2/P3 and fresh run IDs. P3 is
still a 511-1025-token boundary/readout diagnostic, not proof of improved long-
context reasoning. A later 4K/8K/16K preservation and distributed-evidence study
needs a separate registration, qualified precision/backend, actual source assets,
reviewed data and an independent endpoint/power plan before execution. No such
run or approval is created here, and an all-zero/readout-limited endpoint cannot
be presented as successful compatibility.

Record executed test receipts and their limitations in a separate evidence-only
commit. This source/documentation commit contains no performance result.
