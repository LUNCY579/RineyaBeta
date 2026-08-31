# CRCC v1 pre-run execution amendment

**Recorded before model measurements**  
**Preregistration head:** `242613d1b4b5e0a5a4a7154c60337115ca394bd7`  
**Scope:** execution/provenance completion only; no hypothesis, threshold, data, or membership rule changes

This amendment closes the operational details needed to execute the intact baseline and
28 one-layer bypass scan from the CRCC v1 preregistration. At the time this amendment is
committed, no intact-baseline output, bypass output, CRCC membership, recovery result,
long-context result, or sealed result has been observed.

## 1. Official checkpoint pin

The local path `Qwen3-0.6B/` is materialized from:

- repository: `Qwen/Qwen3-0.6B`
- immutable revision: `c916fa4defd319b7d4e4da17604ca7338f4d99f5`
- expected coordinate system: 28 decoder layers

The execution passes `--hash-weights`, so the run manifest records SHA-256 hashes for
checkpoint weight files as well as tokenizer/config metadata. The model repository,
revision, and expected depth are also recorded in `configs/critical_core_scan_v1.yaml`.
This adds provenance; it does not alter model weights or the intervention.

## 2. Runtime pin and resolved precision

The GitHub Actions execution harness pins:

- Python `3.11`
- PyTorch `2.8.0` CPU wheel
- Transformers `4.57.1`
- Accelerate `1.10.1`
- PyYAML `6.0.2`

The preregistered scanner already resolves CPU execution to `float32` even though the
portable config requests `bfloat16`; this behavior was present at the preregistration
head. Every intact and bypass condition uses the same resolved CPU precision and the
run manifest records the device plus software versions. This first discovery package is
therefore a CPU/FP32, baseline-relative structural scan. It is not silently represented
as a CUDA/BF16 run.

## 3. Sharded execution equivalence

The scan is distributed only to fit bounded hosted-runner time:

1. one job evaluates the intact baseline once and layer 0;
2. the exact baseline and run manifest are reused byte-for-byte by seven disjoint layer
   shards covering layers 1–27;
3. each layer still uses the unmodified one-layer bypass implementation and the same 32
   deterministic diagnostic cases;
4. the assembler refuses missing, duplicate, out-of-range, mismatched-case, mismatched-
   config, mismatched-source-hash, or unhashed-checkpoint evidence;
5. the assembler recomputes the preregistered overall and per-source classifiers from
   the raw summaries and requires exact coverage of layers 0–27.

No shard changes the classifier, thresholds, minimum two-source support, manual-layer
policy, generation settings, prompt construction, or NLL calculation.

## 4. Reproducible evidence hash

Raw condition files include elapsed wall-clock time, which is integrity metadata rather
than a scientific measurement. The committed `scan_summary.json` therefore records:

- ordinary file SHA-256 hashes for raw-artifact integrity; and
- a stable scientific `summary_sha256` computed from measured summaries,
  classifications, semantic condition hashes, model identity, and frozen configuration,
  while excluding output paths, raw file hashes, and timestamps.

The model-light verifier proves that the stable summary hash is unchanged across two
assemblies of identical synthetic measurements and that incomplete 28-layer coverage is
rejected.

## 5. Evidence and branch-write policy

Raw case-level outputs remain under `runs/critical_core/` and are uploaded as a private
GitHub Actions artifact. Only the small audited package is committed under:

`experiments/catastrophic_removal_core/results/discovery_v1/`

Before committing that package, the workflow verifies that the remote branch still
points to the exact execution commit. It refuses to overwrite a branch that advanced
while the scan was running.

## 6. Evaluation boundary remains closed

This execution opens only the two training-corpus diagnostic paths already frozen in
the scan config. It does **not** run:

- `chat_dev` or `rag_dev`;
- matched recovery-arm design or training;
- any donor merge;
- the historical 64-row long-context showcase; or
- any sealed evaluation asset.

The evidence manifest must state that the historical 64-row evaluator is not yet
canonically registered or executed and that no result in this package is comparable to
Halo's historical `30/64`. A Halo-comparable claim remains blocked until the exact 64
cases, dataset hash, prompt/context construction, tokenizer, generation settings,
scorer, loading behavior, and known Qwen/Halo anchors are registered and reproduced.

## 7. G0 outcome policy

The automatic membership is frozen only when every preregistered G0 condition passes,
including complete 28-layer coverage, finite intact reference NLL, core depth 1–14,
absence of manual promotion, required support in both diagnostic sources, and hashed
checkpoint weights. If G0 fails, the workflow records `DISCOVERY_COMPLETE__G0_STOP` and
must not design a convenient substitute core by eye.
