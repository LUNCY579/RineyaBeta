# RGE-PEFT Third-Party Notices

## NVIDIA NeMo Emerging Optimizers

`scripts/train/optim/muon.py` contains an adapted, reduced implementation of
Muon update semantics derived from NVIDIA NeMo **Emerging Optimizers**.

- Upstream project: `NVIDIA-NeMo/Emerging-Optimizers`
- Pinned reference commit: `02b6ec843b392475a9dd29fc68dbdb27c48bea4c`
- Relevant upstream files:
  - `emerging_optimizers/orthogonalized_optimizers/muon.py`
  - `emerging_optimizers/orthogonalized_optimizers/muon_utils.py`
  - `emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py`
- Upstream copyright: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
- License: Apache License 2.0
- Included license copy: `third_party/licenses/Apache-2.0.txt`

The Rineya adaptation is materially narrowed and modified for deterministic,
single-device, FP32 LoRA optimizer experiments. It excludes distributed paths,
Triton kernels, tensor parallelism, Hyperball, reduced-precision optimizer math,
and non-matrix routing. The adapted source carries prominent SPDX copyright and
license notices.

No NVIDIA trademark or endorsement is implied.

The pinned commit and license were independently re-verified on 2026-08-30
before local materialization. The Rineya copy adds fail-loud finite-value and
strict-FP32 checks beyond the narrowed upstream semantics.
