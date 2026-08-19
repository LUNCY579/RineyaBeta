# Windows notes

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
$env:PYTHONHASHSEED = "0"
$env:RINEYA_DATASET_ROOT = "C:\data\rineya_sequence_independence_dataset_v1"
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Launchers

PowerShell equivalents of every shell script live in `scripts\*.ps1`:

```powershell
.\scripts\validate.ps1
.\scripts\train_full.ps1  -Seed 1
.\scripts\train_split.ps1 -Seed 1
.\scripts\evaluate.ps1
.\scripts\compare.ps1
.\scripts\smoke.ps1
```

## Known differences

| Topic | Note |
|---|---|
| Line endings | all JSONL is written with `newline="\n"` explicitly, so hashes match across platforms. Set `git config core.autocrlf false` in this repo. |
| Long paths | run directories nest a few levels; enable long paths (`LongPathsEnabled=1`) if you use deep output roots. |
| bitsandbytes | 4-bit/8-bit QLoRA support on Windows is unofficial. If `model.quantization: 4bit` fails, use `none` and reduce `per_device_batch_size` — but use the **same** setting on both arms. |
| DeepSpeed | not supported on native Windows. Use WSL2 or the Docker image. |
| FSDP | requires a working NCCL/Gloo backend; on Windows use Gloo (CPU) only for smoke tests. |
| Symlinks | the HF cache may need Developer Mode enabled, or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. |
| `timeout` | not available; the PowerShell scripts do not rely on it. |
| Paths in configs | forward slashes work everywhere; prefer them over escaped backslashes in YAML. |

## Multi-GPU on Windows

```powershell
accelerate launch --config_file configs/accelerate/multi_gpu.yaml `
  -m rineya_lab.cli train -c configs/train_split_stateful.yaml
```

Backtick is the PowerShell line continuation. Set `$env:CUDA_VISIBLE_DEVICES`
before launching to restrict devices.

## Verify the install

```powershell
rineya-lab info
rineya-lab smoke --dataset-root $env:RINEYA_DATASET_ROOT
pytest -q
```
