# Reproducibility

## What every run records

`run_manifest.json` (training) and `eval_report.json` (evaluation) contain:

- the **fully resolved** config and its SHA-256 fingerprint — not the file it
  came from, so a later edit to the YAML cannot silently change what a past run
  meant;
- model and tokenizer identifiers, revision, precision, quantization;
- LoRA settings, trainable/total parameter counts;
- dataset paths, integrity status and per-file SHA-256;
- the released dataset stats and validation report as found on disk;
- random seeds and what was actually seeded (`python`, `numpy`, `torch`);
- the exact command line (`sys.argv`) and the software/hardware environment,
  including package versions and git commit/dirty state;
- the view plan: order mode, per-epoch row counts, shuffle salts, filters;
- order diagnostics: adjacent-same-problem and adjacent-consecutive-step counts;
- exposure: records, unique problems, input/target/padded tokens, supervised
  positions, optimizer updates, micro-batches, truncation counts;
- checkpoint locations, stop reason, wall clock, throughput, peak VRAM;
- generation settings, raw generations, parsed outputs, per-example score records
  and aggregate reports.

## Sources of determinism

| Mechanism | Why |
|---|---|
| SHA-256 salted sort keys for every shuffle | identical order on any machine, Python version or process; `random.shuffle` is not portable across versions |
| `derive_seed(base, *tags)` | independent but reproducible child seeds |
| `seed_everything` | seeds python/numpy/torch, sets `CUBLAS_WORKSPACE_CONFIG`, requests deterministic algorithms |
| Greedy decoding by default | no sampling variance in the primary metric |
| Custom xorshift RNG in `analysis.stats` | bootstrap/randomization independent of numpy global state |
| `problem_ids_sha256` recorded per evaluation | detects arms evaluated on different sets |
| Atomic manifest writes | an interrupted run never leaves a half-written record that reads as complete |

Set `PYTHONHASHSEED=0` anyway. Nothing here depends on builtin hashing, but the
runbook surfaces a warning if it is unset so extensions do not introduce a
dependency silently.

## Exact resume

`training_state.json` stores epoch, position inside the planned epoch, global
micro-step, optimizer-update count and the exposure account. Resume restores the
optimizer and scheduler state and skips forward *without changing the ordering*,
so a resumed run sees the same examples in the same order as an uninterrupted one.

## Residual non-determinism

Honest list of what is *not* bit-reproducible:

1. **GPU kernels.** Non-deterministic reductions exist even with
   `use_deterministic_algorithms(warn_only=True)`. Different GPU models produce
   different floating-point results.
2. **Mixed precision.** bf16/fp16 accumulate differently across hardware.
3. **Data-parallel batch composition.** Changing the world size changes batch
   boundaries and therefore gradients.
4. **External model downloads.** Pin `model.revision`; a moved tag changes the
   base checkpoint.
5. **Library versions.** Recorded in the manifest, not pinned by the package.
   Use the Docker image for a fixed stack.

Because of these, run at least three seeds and report variance across them. A
difference smaller than the seed-to-seed spread is not a result.

## Reproducing a past run

```bash
python -c "import json;print(json.load(open('runs/X/run_manifest.json'))['config_fingerprint'])"
python -c "
import json
from rineya_lab.config import config_from_dict
cfg = config_from_dict(json.load(open('runs/X/run_manifest.json'))['resolved_config'])
from rineya_lab.training.loop import train
train(cfg)
"
```

Then compare `config_fingerprint` values. Identical fingerprint + identical
dataset hashes + identical environment ⇒ the run is reproduced up to the residual
non-determinism above.

## Docker

```bash
docker build -t rineya-lab .
docker run --rm -v "$PWD/../rineya_sequence_independence_dataset_v1:/data:ro" \
  -e RINEYA_DATASET_ROOT=/data rineya-lab rineya-lab validate-data --splits dev
```

The image pins the Python version and installs from `pyproject.toml`. It does not
pin CUDA; match the host driver when building the GPU variant.
