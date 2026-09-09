# Composition and query-conditioned re-reading research

This branch adds the isolated programme in
[experiments/composition_qrr_v1](experiments/composition_qrr_v1/README.md).
Start there for the current stage table, runnable commands and explicit blockers.

It preserves the existing experiment history, training code, checkpoints and
separate HLGA/Cornerstone branches. The first target is the actual R2 checkpoint,
not a shape-similar Supa substitute. Current results are CPU software/tensor
qualification only, not pretrained-model efficacy. Training and sealed evaluation
remain closed. The full/cached frozen runner is implemented but requires an exact
reviewed local asset/runtime contract before it can run.
