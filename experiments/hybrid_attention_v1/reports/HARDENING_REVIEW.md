# HLGA-1 reviewed hardening evidence — 2026-09-08

## Remote source and scope

Infrastructure commit: `5c77c64fea4ba4f0ceaff8a2b894a8944d300da2`.
Reviewed parent: `f36425368b098d41fc4802503ba1f43f372050da`.
Branch: `research/hybrid-local-global-attention-v1`.
Canonical main observed unchanged at `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`.

The earlier preserved package was useful but stale: it targeted `6770e67...` and
eight arms. The live branch already contained ten arms and authoritative workload
accounting in `study.py`. This increment adapted only the missing declaration
auditor, reused the live contrasts/planner, and fixed execution/evidence defects.
It did not reapply the old protocol, old publisher script, or old test receipt.

Nine source/documentation paths changed, all within this experiment. All nine
uploaded blob SHAs matched the exact locally tested bytes before the commit was
created. The branch update was a normal fast-forward, not a force push. Main,
other branches/tags, checkpoint/tokenizer assets, shared trainers, datasets and
historical receipts were not changed by these actions.

## Findings and disposition

1. **Nonfinite generation was silently accepted.** A tiny upstream fixture with
   NaN output weights returned token IDs `[0, 0]` rather than failing. Streaming
   logits now must be finite before token selection, including intermediate
   prefill chunks. Nonfinite prefill/decode regression tests pass.
2. **Early streaming errors retained request state.** An invalid append raised
   `ValueError` but left cached position 12 in the upstream fixture. All forward
   failure locations, including validation, embeddings and RoPE, now clear state.
   A new generation also resets before validating arguments.
3. **Failure diagnostics could themselves overflow.** Finite opposite-sign FP32
   extrema overflowed the original subtraction; error magnitudes now use FP64
   and remain JSON-serializable. Nonfinite receipt payloads are rejected.
4. **Source/contracts and outputs needed stronger binding.** Root Markdown now
   contributes to the source fingerprint. Successful runs verify that source
   identity stayed unchanged and bind the accompanying JSON artifact membership,
   sizes and hashes. P3 requires its P2 bundle to match. These are integrity
   checks, not signatures, proof of real model identity or authorization.
5. **Required native verification needed stricter pass semantics.** Empty or
   skipped test collections cannot satisfy `--require-hf`.
6. **Comparison metadata needed treatment and budget verification.** The new
   auditor checks the full executed spec against its arm name, canonical geometry,
   paired identities and exposure, teacher/loss/mask/selection consistency, and
   complete training-compute declarations. It rejects forward-only estimates and
   shared/fixed-order time as matched training compute. It never opens a stage.

No attention topology, gate formula, healthy decoding policy, scoring rule, case
order, prompt content, primary contrast or study budget was changed. Evidence
spans/gaps are additional metadata, not a new diagnostic corpus or endpoint.

## Executed local checks

`hardening_local_receipt.json` and `hardening_local_tests.log` record **38/38 new
tests passed, zero skips/failures/errors**, with Python 3.13.5 and torch 2.10.0+cpu.
Transformers was absent locally. This was a selected, byte-verified module
snapshot, not a complete checkout or a local full-native qualification.

Separate before/after checks used the same tiny random eight-layer fixture:
40 arm/backend/block combinations produced identical token IDs, cache metrics and
attention counters. All 72 generated diagnostic cases preserved every preexisting
field, including input token IDs, prompt hashes and answers; only four position
metadata fields were added. This is regression evidence, not quality evidence.
The local receipt records the exact published file hashes and these checks.

## Executed full native CI

GitHub Actions run **34288940094**, job **102270888757**, completed successfully
for the infrastructure commit above:

- https://github.com/LUNCY579/RineyaBeta/actions/runs/34288940094
- Command: `python -m experiments.hybrid_attention_v1.verify --run-id ci-native --require-hf`
- **95/95 tests passed; zero skipped, failures or errors.**
- Native `Qwen3ForCausalLM` random-model integration actually ran and passed.
- Python 3.11.16, torch 2.6.0+cpu, Transformers 4.57.1, safetensors 0.8.0,
  tokenizers 0.22.2; FP32 CPU, one Torch thread.
- Source unchanged during verification: true.
- Source fingerprint: `50b5ef813f72efe3e66689899588c083844debbe81b8b772ddc9770fbb567060`.
- Receipt payload digest: `fc72145a0b28e49f3def7f564528c3ae2e690b1fcd9001f224b588c9aca3d3ba`.

Artifact **10080587385** was downloaded and its archive SHA256 verified as
`c21985d391d59e2fdc703ddac6e4ac177cc4ea220a51077c90a13e69c3f880ef`.
`hardening_ci_receipt.json` and `hardening_ci_environment.txt` preserve the exact
extracted bytes. The receipt's canonical JSON payload digest was independently
recomputed and matched the logged digest above. These files are evidence for
this tested source, not manually invented replacement receipts.

The 95 tests include the preexisting toy gate optimizer test. No pretrained R2
training, real Rineya quality generation, GPU benchmark or sealed evaluation ran.
The shape probe uses synthetic tensors with R2 dimensions; its resident-cache
measurement is not an actual Rineya request or whole-process memory measurement.

The workflow also emitted Node.js action deprecation warnings. It succeeded under
the runner's Node 24 behavior; those warnings are infrastructure maintenance items,
not a model failure or grounds to enable insecure Node compatibility. The workflow
and dependency pins were not changed in this increment.

## What the evidence permits

This is a justified infrastructure hardening increment. It establishes that the
reviewed regression checks and full pinned native CPU suite pass; it does not
prove that no bugs remain or that every Rineya-family checkpoint is compatible.
The declaration auditor checks metadata consistency, not the authenticity of the
profiler, source lineage or isolation claims it references.

The next executable model step remains an actual local R2 asset audit and fresh
P2 mechanical qualification on the intended device/backend. Old receipts cannot
cover the new fingerprint. Only a passing current P2 bundle permits the existing
bounded P3 open diagnostic. Its 511–1025-token probes are boundary/readout tests,
not evidence of better 4K/8K/16K reasoning. A zero-score or UNKNOWN-loop tie is
readout-limited, not a successful noninferiority result.

Training P4 and confirmation P5 remain closed. Later recovery training must keep
the trained all-global gate control, distinguish equal-data and equal-total-
compute studies, review data provenance/splits and preregister selection and
independent evaluation. Other family checkpoints, precisions, kernels and longer
contexts need their own qualified registration. No model promotion, efficiency,
context-extension or long-context quality claim is licensed by this commit.

This evidence-only addition is under `reports/`; no executable or root-contract
bytes change, so the tested source fingerprint remains applicable. Historical
receipts and negative results remain intact.

> **SUPERSEDED (2026-09-09).** The sentence above was true when written but is
> no longer: commits after it rewrote `runtime.py`, `run.py`, `attention.py`,
> `diagnostics.py` and the tests, so this report's source fingerprint does NOT
> describe current code and its receipts cannot gate anything. Every receipt
> preserved before that point also carries
> `real_rineya_checkpoint_loaded: false`. See
> `AGENT_REAL_MODEL_QUALIFICATION.md` for the first real-R2 qualification and
> the current code identity.
