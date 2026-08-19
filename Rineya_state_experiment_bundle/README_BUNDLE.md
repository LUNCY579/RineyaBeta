# Rineya State Sufficiency experiment bundle

Everything needed to run the experiment **except the model weights**, which are
excluded deliberately: the checkpoint is a published artifact and should be
fetched from its source rather than copied around.

```
rineya_state_sufficiency_lab/              the pipeline (code, configs, docs, tests, scripts)
rineya_sequence_independence_dataset_v1/   48,000 problems / 189,922 transitions
```

## Setup

```bash
cd rineya_state_sufficiency_lab
pip install -e ".[all]"
export PYTHONHASHSEED=0
export RINEYA_DATASET_ROOT="$(cd ../rineya_sequence_independence_dataset_v1 && pwd)"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='../qwen3-0.6b',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
"
```

The configs expect the checkpoint at `../qwen3-0.6b`. Anywhere else works via
`--set model.name_or_path=/abs/path`.

If the checkpoint already exists elsewhere — say one level further out, shared
across sibling projects — point the eight real-model seam tests at it instead of
copying 1.5 GB:

```bash
export RINEYA_QWEN_DIR=/abs/path/to/Qwen3-0.6B
```

Those tests are the only coverage of the real tokenizer, chat template,
left-padding, LoRA attachment and batched-generation paths — the ones the GPU run
actually takes. They **skip silently** when the checkpoint is not found, so
confirm `pytest -q` reports `266 passed` rather than `258 passed, 8 skipped`
before trusting a green suite. `../qwen3-0.6b`, `../Qwen3-0.6B` and the same two
names one directory further out are discovered automatically.

## Verify the bundle arrived intact

```bash
python -m rineya_lab.cli validate-data --splits dev test   # expect status: PASS
python -m rineya_lab.cli smoke --all                       # expect status: PASS
pytest -q                                                  # 266 tests, 0 skipped once found
```

`validate-data` re-derives the FULL/SPLIT pairing and checks file hashes against
the dataset's own `sha256sums.txt`, so it doubles as a transfer-integrity check.

## Running it

**On a CUDA GPU** (whole design in about a day):

```bash
scripts/run_gpu_rehearsal.sh          # ~45 min, prints a go/no-go verdict
scripts/run_gpu_experiment.sh 1 2 3   # full design, DEV only
scripts/run_gpu_test_phase.sh 1 2 3   # refuses until the margin is frozen
```

**On a CPU-only host** (measured on an 8 GB Apple Silicon Mac: 0.082 train rows/s,
32.2 generated tok/s at batch 8):

```bash
STAGE=calibrate scripts/run_mac_pilot.sh       # ~10 min, prints an hours-per-size table
TRAIN_PROBLEMS=1000 scripts/run_mac_pilot.sh   # ~12 h to the G9 go/no-go, ~37 h to seed 1
SEED=2 TRAIN_PROBLEMS=1000 scripts/run_mac_pilot.sh
SEED=3 TRAIN_PROBLEMS=1000 scripts/run_mac_pilot.sh
python -m rineya_lab.cli aggregate runs/pilot_comparison* --output-dir runs/agg_pilot
```

The pilot reduces **training** scale only and evaluates on the full 2,400-problem
DEV split, because statistical power comes from paired problems evaluated, not
from training size. It is resumable stage by stage: re-running after a crash,
reboot or Ctrl-C is always safe.

Read `docs/READINESS_ASSESSMENT.md` first, then `docs/RUNBOOK.md`.

## Before reporting anything

Fill `docs/PREREGISTRATION.md` sections 3 (seeds), 6 (non-inferiority margin) and
8b (learnability floor). Comparison reports carry a warning until the margin is
frozen, and the TEST-phase script refuses to run.

## Not included

* model weights (`model/`, `qwen3-0.6b/`)
* `runs/` — all prior runs were smoke tests, probes and calibration, none of them results
* `dist/` — build output

## Status

No adapter has been trained on this dataset with a real checkpoint. This bundle is
a validated *pipeline*, not an experiment that has been run.
