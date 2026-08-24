# SGQR Implementation Readiness

## Current status

`NOT_READY_FOR_SCIENTIFIC_EXPERIMENTS`

## What has been verified in the reference implementation

Synthetic/reference checks cover:

- exact INT4 nibble packing;
- INT8 storage;
- row-wise quantization error bounds;
- frozen remnant buffers;
- additive group reconstruction;
- selected group execution;
- token-specific routing;
- exact deleted and all-on wrapper behavior;
- static and dynamic top-k significance;
- state-dict reload;
- utility sign;
- exposure-normalized significance;
- Recovery Ratio.

## What remains before scientific interpretation

The following require execution on the real Rineya checkpoint and GPU:

- dense protocol reproduction;
- correct target-layer intervention in the installed Transformers version;
- all-off versus physical deletion on real prompts;
- all-on versus complete quantized remnant on real prompts;
- candidate-layer scan;
- INT8/INT4 donor viability;
- utility calibration;
- significance-versus-random comparison;
- CUDA memory accounting;
- synchronized latency;
- deployment serialization/reload with the actual base checkpoint.

The status may change to `READY_FOR_SCIENTIFIC_EXPERIMENTS` only after these
real-model invariants pass and are recorded.
