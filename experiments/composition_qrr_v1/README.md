# Composition adjudication and QRR-1

This isolated programme asks whether operator composition can improve through a
better instrument, conventional adaptation, or genuinely useful sequential reads.
It does not replace the released architecture, modify historical evidence, or
claim that all 0.6B models share a capacity limit.

Branch: `research/composition-qrr-v1`. Base: RineyaBeta
`3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`. Only this new directory and the new root
routing note are added. Main, Cornerstone and HLGA remain separate.

## What exists now

| Stage | Implemented | Evidence boundary |
|---|---|---|
| E00 | Byte inventory, LFS/symlink/mutation rejection, pinned source ledger | Real R2/Supa weight inventory not run here |
| E01 | Separate R2-only local loader, full/cached qualification, frozen runner | No transformers or model weights in the build environment; native unrun |
| E02 | Six operators, two renderers, paired worlds, independent list/map oracles, direct/oracle-evidence/oracle-binding prompts, scorer | Open synthetic instrument only; no language-model score |
| E03/E04 | Gates and work orders for microfit and conventional adaptation | No optimizer implementation or authorization |
| E05 | QRR tensor mechanism, 1/2/4 tied reads, untied-two control, exact-parameter MLP, logit-residual seam | Not installed in HF generate/cache; native QRR unqualified |
| E06 onward | Separate HLGA, future indexer/scaling/recovery plan | Deferred; no imported performance claims |

See `protocol.json`, `DESIGN.md`, `SOURCES.md` and `evidence/initial_cpu/`.
The source digest includes Python source/tests plus the protocol, not evidence
receipts. A new source/runtime/checkpoint invalidates the corresponding native
receipt. Receipts are provenance records, not cryptographic proof that an arbitrary
JSON file was honestly produced; review their execution logs and provenance.

## Run the CPU qualification

From the repository root, in an environment with PyTorch:

```bash
python -m unittest discover -s experiments/composition_qrr_v1/tests -v
python -m experiments.composition_qrr_v1 source-id
python -m experiments.composition_qrr_v1 arms
python -m experiments.composition_qrr_v1 build-bank --worlds 8 --seed 17 --out /tmp/qrr-open-dev.json
python -m experiments.composition_qrr_v1 audit-bank /tmp/qrr-open-dev.json
```

The recorded build used Python 3.13.5 and torch 2.10.0+cpu. This is an observed CPU
fixture environment, NOT a required CUDA environment or proof about another
PyTorch/transformers pair. Native versions must be reviewed and bound explicitly.
All writes use exclusive creation; choose new paths for new runs. No full-repository
regression suite or legacy claim verifier is represented by this focused suite.

## First real-model action

Inventory the actual R2 directory; a config, LFS pointer, differently named Supa or
random model is not a substitute:

```bash
python -m experiments.composition_qrr_v1 inventory artifacts/rineya-core-en32k-28L --out /tmp/r2-assets.json
```

Review the byte inventory and actual lineage independently. Copy
`native_contract.template.json` to a new run location, fill the independently
verified assets, source digest, exact Python/torch/transformers/CUDA/GPU identity,
and reviewed precision, then record the reviewer and frozen-only status. Merely
observing hashes is not approval. The untouched template deliberately fails.
R2's committed config stores BF16; a different declared realization needs separate
qualification. Supa-A1.1's FP16 requirement does not override R2's identity.

```bash
python -m experiments.composition_qrr_v1 qualify-native --checkpoint artifacts/rineya-core-en32k-28L --contract /tmp/r2-reviewed.json --out /tmp/r2-native-receipt.json
python -m experiments.composition_qrr_v1 diagnose-native --checkpoint artifacts/rineya-core-en32k-28L --contract /tmp/r2-reviewed.json --receipt /tmp/r2-native-receipt.json --bank /tmp/qrr-open-dev.json --out /tmp/qrr-runs --run-id r2-open-dev-001
```

Qualification forces four mechanical steps even through EOS to exercise cached
decode; these probes are not task results. Actual diagnostics respect registered
EOS and a 24-token cap. No truncation, implicit CPU fallback, remote download,
custom remote code, inherited calibration curve, or checkpoint write is allowed.
All direct/evidence/binding prompts are token-budget checked before task inference.
The default bank has 96 cases and the ladder has 288 generations; its larger hard
cap is 64 worlds / 768 cases / 2304 generations. Runtime contracts may only lower
those limits. Timings are descriptive, not speed comparisons.

## Interpretation and next gates

Literal exactness, declared format equivalence, wrong abstention and wrong-cue
substitution are separate metrics. Oracle-assisted results must never be reported
as deployable improvements. `analysis.compare` pairs by case and resamples worlds;
exact McNemar is exposed only with one binary observation per world. Its empirical
interval is not a zero-harm bound or a confirmatory multiple-comparison procedure.

Length/position banks preserve the same worlds and answers. Their padding counts
are NOT promised token lengths: the native runner records actual token counts and
rejects over-budget prompts. Cross-length pairing currently uses cluster/operator/
renderer keys; the within-bank comparison helper intentionally rejects different
bank identities. Build a separately reviewed cross-bank contrast before pooling.

Run the frozen evidence/binding ladder first. Then separately register a 32-64
training-world microfit, a conventional adapter/MLP/LoRA screen, and only then a
QRR efficacy contrast against the strongest simple baseline. Required additions
include held-out renderers/programs, model-extract-then-solve, wider trainable-path
controls, resource pilot, seed/preservation policy and independent test-bank
registration. No training or sealed-evaluation command exists in this revision.

Keep HLGA recovery on its own branch. Reduced context cost does not by itself
explain or repair failure on short operator problems. No new report here overturns
any preserved negative result or claims architecture novelty.
