# HLGA-1 revision 2: infrastructure evidence review

Tested infrastructure commit: `6cc101ac90aec570ecc9e4fd8a004fe8f3c46e26`.
Parent implementation preserved: `6770e67b471f4821004175d5d39a228f079b28f1`.
This report is an evidence-only addition; it does not alter executable sources,
protocol, model assets, datasets, or another branch's approvals.

## What changed

The branch already contained eight-arm attention/runtime infrastructure and its
historical receipts. This increment adds A8 all-local and A9 equal-global-count
frontloaded controls, fixes A3 versus A0 as the primary frozen comparison, adds
direct strict/trimmed and answerability-stratified contrasts, and records exact
submitted attention rectangles separately from semantically visible pairs.
The planner includes QKV/O, MLP, LM-head and gate matrix work, rather than treating
attention-only FLOPs as full-model cost. Per-case generated-length accounting is
checked against observed execution shapes and compact resident KV storage.

The CLI locks the primary R2 shape, limits blocks/cases/CUDA allocator requests,
requires seven current-checkpoint mechanical checks including both new controls,
and rejects stale study versions and unchanged-checkpoint omissions. NaN/Inf
qualification failures remain serializable. The diagnostic has at most ten arms,
72 prompts per arm and 24 generated tokens per prompt. Training and sealed
commands still refuse unconditionally.

Future gate-only token-matched and compute-matched plans are distinct and CLOSED.
Three seeds, 64 token-matched updates and a 128-update compute-match cap are
planning bounds, not a training receipt. Reviewed data provenance, optimizer,
supervision masks, whole-update profiling and independent confirmation remain
required. See [the contract](../COMPATIBILITY_CONTRACT.md).

## Locally executed evidence

`python -m unittest experiments.hybrid_attention_v1.tests.test_study -v`
passed **25/25 tests, zero failures, errors or skips**. The accompanying
[receipt](bounded_study_local_receipt.json) records the exact source hashes,
imported dependency Git blobs, environment and limitations; the
[log](bounded_study_local_tests.log) retains individual test outcomes.
All eight changed-file Git blob hashes were checked against the remote commit
before the branch was advanced. `compileall`, `run plan` and the analytical
`study` CLI also completed successfully locally.

The local environment was Python 3.13.5, torch 2.10.0+cpu, one thread, without
Transformers or CUDA. Local tests used small randomly initialized Qwen3-shaped
fixtures, including distinct eight-layer layouts. They covered semantic masks,
submitted rectangles, generated-token/cache off-by-one boundaries, early EOS,
all ten arms, corrupt counters/cache, budget arithmetic, direct contrasts,
qualification and non-authorizing failure paths.

**This is a local subset receipt, not the complete inherited test suite.**
`test_core.py` and `verify.py` were not executed locally, and package initializer
stubs were used only for local imports. No full-tree code fingerprint is claimed
from that local materialization. The remote inherited files were left untouched.

## Independently completed native-HF CI

GitHub Actions run **34279792647**, job **102241524827**, checked out exactly
`6cc101ac90aec570ecc9e4fd8a004fe8f3c46e26` and completed successfully on
2026-09-08 at 21:18:49 UTC. Its command was
`python -m experiments.hybrid_attention_v1.verify --run-id ci-native --require-hf`.
All **57 tests passed with zero skips, failures or errors**, including the
inherited 32 tests and the 25 new tests. The native random-Qwen3 integration test
ran successfully; this was not a substitute mock or a real Rineya checkpoint.

The CI environment was Python 3.11.16, torch 2.6.0+cpu, Transformers 4.57.1,
safetensors 0.8.0 and tokenizers 0.22.2, FP32, CPU, one thread. The exact
[CI receipt payload](bounded_study_ci_receipt.json) was transcribed from the
returned decoded job log and verified against its logged SHA-256:
`5c0ba0e735b76b6fbda10201e08fc7aace3ab438240c693a1f6832c39fdf6b73`.
Its full experiment source fingerprint is
`dfe472ecf6042e6237f85a4ded0fc562d25efe3178ee3b29f03da3c755c10034`.
This evidence-only report does not change that executable/protocol fingerprint.

The synthetic R2-dimension cache probe also passed: resident KV 146,857,984 bytes
matched its analytical budget at 1025 tokens. It did not load model weights or
measure whole-process peak memory. Neither this tensor result nor test elapsed
time establishes a model speedup or quality/resource Pareto improvement.

Source run: https://github.com/LUNCY579/RineyaBeta/actions/runs/34279792647 .
The workflow uploaded its original receipt and environment file as artifact
`hlga1-cpu-native-6cc101ac90aec570ecc9e4fd8a004fe8f3c46e26`, ID 10077162048.
Historical 32-test CI evidence remains valid for its own parent revision only.

## What is not established

No real Rineya checkpoint was loaded, trained or quality-tested locally. No
GPU benchmark, preservation result, held-out evaluation, family-wide transfer,
context extension or performance improvement is established by these tests.
The bound 72-case synthetic endpoint cannot establish broad noninferiority;
equality at zero or repeated UNKNOWN is readout-limited/inconclusive.
All diagnostic timings remain non-causal. No production model is exported.

The next evidence-bearing action is local real-R2 identity audit and P2
mechanical qualification on the intended runtime. Only a matching
passing real-checkpoint receipt permits the open P3 diagnostic. P4/P5 remain
closed. Infrastructure commits and test receipts must not be presented as model
performance claims.
