# RineyaBeta — hybrid attention research branch

This branch adds **HLGA-1: periodic local/global attention with optional persistent
sink tokens and identity-centered head gates**. It does not replace the released
Rineya architecture or revise earlier experimental verdicts.

**Start here:** [experiment and runbook](experiments/hybrid_attention_v1/README.md).
See the [registered protocol](experiments/hybrid_attention_v1/protocol.json) and
[implementation review](experiments/hybrid_attention_v1/reports/IMPLEMENTATION_REVIEW.md)
for precisely what has run and what has not.

Branch: `research/hybrid-local-global-attention-v1`  
Base: `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
Initial target: **R2**, `artifacts/rineya-core-en32k-28L`, with actual local weights
required. A config file, a random model, and a Supa checkpoint are not substitutes.

Current evidence is **CPU fixture qualification**, not a Rineya quality result.
Real-checkpoint parity, diagnostic generations, training and GPU performance have
not been measured in the committed local receipt. Training and sealed evaluation
remain unavailable in this experiment's CLI.

## Existing program

The [historical routing map](docs/README.md) and
[canonical experiment history](docs/EXPERIMENT_PROGRAM_AUDIT.md) remain unchanged.
In particular, the earlier [attention-shape results](experiments/attention_shape/README.md)
are necessary background: naive windowing already failed in that tested stack.
This branch tests specific proposed remedies rather than treating that failure as
never having happened. It leaves `scripts/train/`, checkpoints, corpora, frozen
suites, and the Cornerstone branch untouched.
