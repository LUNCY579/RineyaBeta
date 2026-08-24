# SGQR Experiment Log

## 2026-08-24 — Branch initialization

- Repository: `LUNCY579/RineyaBeta`
- Branch: `research/sgqr-core-quantized-remnant`
- Base: `r2-healing-harness`
- Base SHA: `1664f54de0bbdfde45e311b75bbf3038dbfffe42`
- Purpose: isolate the quantized-remnant and significance mechanisms from
  confidence gating.

### Reference validation

- Python compilation: passed for the new SGQR files.
- `python scripts/verify/verify_sgqr.py`: 14/14 checks passed in the construction
  environment.
- `bash -n experiments/sgqr/run_reference.sh`: passed.

### Limit

No Rineya weights or GPU experiment was executed during branch construction.
No quality, recovery, latency, or VRAM claim is made.
