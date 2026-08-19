# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is semantic.

## [1.0.0] — 2026-08-18

Initial release of the standalone lab.

### Training
- FULL-context LoRA training (condition A).
- SPLIT stateful-transition LoRA training (condition C, primary).
- SPLIT next-step-only ablation; ordered, reverse and curriculum order ablations
  (condition B and variants).
- QLoRA (4-bit / 8-bit) configurations; automatic BF16/FP16/FP32 selection.
- Assistant-only loss masking computed exactly, not by marker search.
- Target-preserving truncation with head+tail prompt retention and counted
  truncation events.
- Gradient checkpointing, gradient clipping, cosine/linear/constant schedules.
- Exact checkpoint resume (optimizer, scheduler, epoch position).
- Deterministic seeding with SHA-256-based, machine-independent ordering.
- Per-problem loss weighting (`inverse_transitions`).
- Exposure and token accounting: records, unique problems, input/target/padded
  tokens, supervised positions, optimizer updates, truncations.
- Budget regimes: native epochs, matched optimizer updates, matched target
  tokens, matched source-problem exposure, one-transition-per-problem sampling.
- Hugging Face chat-template support with a portable plain-text fallback.
- Accelerate single-GPU / multi-GPU, FSDP and DeepSpeed ZeRO-2 configurations.
- CPU-only dry-run mode that exercises the whole data path without a model.

### Evaluation
- Protocols: `full_direct`, `direct_answer`, `oracle_transition`,
  `full_history_transition`, `no_state_transition`, `last_state_only`,
  `shuffled_state`, `distractor_state`, `oracle_trajectory`, `generated_rollout`.
- Autonomous rollouts consuming the model's own generated states, with loop,
  stagnation, premature-completion and nontermination detection.
- First-error-depth analysis and the teacher-forcing-versus-free-rollout gap.
- Response-parsing contract: exact/fenced/embedded/repaired JSON, final-answer
  tags, boxed LaTeX, labelled answers, final-line fallback; empty, refusal,
  ambiguous, missing-field, invalid-flag and backend-failure handling.
- Strict (primary) and lenient (diagnostic) scoring tracks; uninterpretable
  responses stay in the denominator.
- Typed deterministic scorers for 21 dataset answer types plus configurable
  regex tasks; configurable tolerances, unit synonyms and scorer mapping.
- Deterministic reference backend with eleven behaviour profiles for offline
  end-to-end testing of the harness.

### Analysis
- Pairing by original problem id, never by expanded transition row.
- Paired cluster bootstrap, exact McNemar, sign-flip randomization.
- Superiority / non-inferiority / equivalence decisions with configurable
  practical-equivalence margins and a not-frozen warning by default.
- Category, trajectory-depth and failure-type stratification with
  Holm-Bonferroni correction.
- Strict-versus-lenient disagreement analysis and deterministic audit samples.
- Validity warnings for reference-backend runs, mismatched evaluation sets and
  split mismatches.

### Reproducibility and documentation
- Every run records the fully resolved config plus fingerprint, model and
  tokenizer identifiers, adapter settings, dataset hashes, seeds, command line,
  software and hardware environment, exposure counts, checkpoints, generation
  settings, raw generations, parsed outputs and per-example scores.
- Preregistration template, metric card, failure taxonomy, runbook, evaluation
  standard, response-parsing contract, dataset assumptions, extension guide,
  reproducibility notes and Windows notes.
- Machine-readable JSON schemas for run manifests, evaluation records,
  evaluation reports and comparison reports, verified against real artifacts.
- Docker image, GitHub Actions CI, Makefile, Unix and PowerShell launchers.
- Strict dataset validator including a causal safeguard against future
  information leaking into an earlier state, plus safe bundle extraction with
  zip-slip, symlink and compression-ratio protections.
