# Query Refinement v1 implementation review

## Saved work and isolation

Implementation commit: `064b9dc01de4b38caedcf101395fba83629dca00`.
Branch: `research/query-refinement-attention-v1`.
Parent: `main@3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`.

This branch adds a bounded Q-only two-read mechanism, an isolated Qwen3 runtime,
matched controls, an executable qualification/smoke/open-DEV pipeline, fresh
procedural diagnostics and CI. It does not merge other research branches. Existing
root subtrees (including scripts, configs, artifacts and documentation) and the
pre-existing experiment subtrees retain their original Git tree hashes.
No original checkpoint, tokenizer, result artifact or sealed evaluation was changed.

## Executed evidence

| Scope | Outcome | What it establishes |
|---|---|---|
| Local isolated test suite, Python 3.13.5 / PyTorch 2.10.0+cpu | 67 passed, 1 skipped | Synthetic numerical, causal, cache, training, provenance and data checks |
| GitHub CI, Python 3.11.16 / PyTorch 2.10.0+cpu / Transformers 4.57.1 | 68 passed, zero skipped | Includes the native HF Qwen3 random-weight oracle and a short optimization test |
| Native CI qualification on a tiny randomly initialized Qwen3 | All checks passed | Zero-update native equivalence and enabled-intervention mechanics |
| Actual pretrained R2 checkpoint qualification | NOT EXECUTED | No R2 compatibility or quality result |
| Registered R2 25-step smoke / three-seed DEV study | NOT EXECUTED | No trained R2 efficacy result |
| Whole-repository historical test suite | NOT EXECUTED | This is an isolated experiment validation, not a legacy regression certification |

CI run: https://github.com/LUNCY579/RineyaBeta/actions/runs/34313773815
Job: `102345609815`, success, completed `2026-09-09T05:10:52Z`.
The workflow artifact includes `qref-junit.xml` and `qref-native.json`.
`ci_review.json` records the run/job/artifact identifiers and artifact ZIP hash.
`native_ci_qualification.json` preserves the JSON printed by the successful CI
step; its source SHA256 manifest matches the locally tested and remotely committed
files. Local execution metadata is in `local_qualification.json`.

For the native random-weight model, all four arms had a maximum zero-update
logit error of exactly 0.0 versus untouched HF eager forward. The largest enabled
full-versus-chunked-cache error was 1.1920928955078125e-07. The tiny fixture's trained
arms have 100 parameters; the registered R2 shape has 2064. Neither number is a
model-quality score. Nonzero finite gradients, causal isolation and frozen-backbone
training were checked; no claim of convergent reasoning follows from these checks.

## Issues prevented during implementation

The implementation distinguishes residual width from total Q-head width; preserves
native QK-Norm and applies RoPE exactly once to each new query; uses explicit
absolute-position visibility rather than ambiguous non-square causal masking;
keeps both reads live in the compute control; and initializes only the up matrix
to zero so learning can start. A stop-gradient control was removed because it is
redundant with a frozen backbone and a single intervention layer.

Nonzero-feedback cache tests, future-value Jacobian tests, active-parameter checks,
checkpoint reload, invalid-input and NaN cache resets, no-truncation token budgets,
exact graph answer oracles and receipt-drift failures are part of the suite.
Measurements remain seed-paired, and three-seed intervals are labeled descriptive.

## Real-checkpoint blocker and next gate

The inspected GitHub directory `artifacts/rineya-core-en32k-28L` at the pinned base
contains LICENSE, config.json, generation_config.json, merges.txt, tokenizer.json,
tokenizer_config.json, trim_manifest.json and vocab.json. It does NOT contain a
model.safetensors or pytorch_model.bin checkpoint. No actual R2 weights were
available in this session's execution environment, and CI intentionally uses no
pretrained downloads. Config/tokenizer metadata cannot substitute for weights.

The next valid operation is `pipeline qualify` on the real local R2 checkpoint,
using the README commands. Only its source-bound passing receipt can open the
25-step R2 smoke, followed by the registered open-DEV comparison. These committed
P0/P1 reports are deliberately NOT accepted as P2/P3 authorization receipts.

## Scope and residual limitations

The full historical-query workspace, cross-token latent recurrence and K/V
modification variants are specified as future work, not claimed as implemented.
The current experiment tests the proposed within-token read/revise/read mechanism.
Runtime qualification is float32, unpadded batch one, original Qwen3 RoPE and
context limits. Mixed precision, quantization, padding/packing, multi-device execution,
optimized GQA kernels, resumable training, general-language regression coverage and
large long-context evaluations need independent work before deployment.
The CUDA memory gate is not an OOM guarantee. Source-bound receipts detect drift,
not malicious forgery. The tiny open graph DEV set is not a statistically powered,
representative or sealed benchmark; a DEV win would justify further study only.

**Conclusion: the implementation and mechanical/native-random tests passed.
No claim that query refinement improves pretrained Rineya is authorized.**
