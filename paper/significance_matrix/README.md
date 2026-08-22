# Observation-Normalized Significance Matrices for Repeated KV-Cache Eviction

This directory contains the audit-corrected, conference-neutral two-column LaTeX
rewrite of `docs/PAPER_DRAFT_SIGNIFICANCE.md`. The original Markdown draft is
preserved unchanged.

## Build

```bash
latexmk -pdf main.tex
```

The source is split into `sections/` so that method, evidence, limitations, and
appendices can be reviewed independently.

## Scope of the rewrite

The paper narrows the contribution to observation-count normalization under
repeated chunked prefill. It does not claim state-of-the-art performance. The
rewrite:

- counts both fp32 score and count state, giving 1.5625% of bf16 KV storage at head dimension 128;
- distinguishes prefix-budget fraction from the actual final resident fraction;
- describes six paired trials as fragile rather than incapable of significance;
- removes the pooled 36 key-depth McNemar claim because outcomes are clustered by key;
- positions novelty against AhaKV, NACL, H2O, SnapKV, and related work;
- labels mixed precision exploratory because it is not implemented end to end.
