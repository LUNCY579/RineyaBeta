# Current Status

## Implemented

- frozen 13-arm registry;
- notice/withdrawal/consolidation schedules;
- loss-only uniform, random, static, state, and state-position responsibility routers;
- interventional token-position significance;
- component-specific residual reconstruction loss;
- Recovery Ratio and Handoff Completeness;
- gate-zero and physical-deletion surgery utilities;
- parameter-frontier and deployment-state checks;
- deterministic CPU reference benchmark;
- paired statistical and mechanistic analysis utilities;
- optional Hugging Face/Qwen integration seams;
- unit, contract, and integration tests;
- path-scoped CI workflow.

## Evidence currently committed

A controlled CPU reference experiment runs all arms on a small residual sequence model and checks the complete measurement pipeline. It is evidence that the experiment is executable and that the required ablations are distinguishable. It is **not** evidence that Rineya has already retained quality after real layer deletion.

## Not yet established

No claim is currently made that:

- PDRR improves a trained Rineya checkpoint;
- PDRR beats Minitron, CoFi, NutePrune, GPP, Neuron Merging, DOTResize, or another external method;
- state-position routing is necessary at Rineya scale;
- causal responsibility actually migrates in Rineya;
- the method generalizes from whole-layer deletion to width, MLP-neuron, attention-head, or KV-head pruning.

Those claims require the local ignored model weights, frozen evaluation payloads, and confirmatory runs described in `RUNBOOK.md`.
