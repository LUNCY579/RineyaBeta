# PDRR Preregistration

Frozen before confirmatory Rineya runs.

## Research question

At the same final compact architecture, data, and optimization budget, does explicit pre-deletion functional-responsibility transfer preserve more quality than deleting a Transformer block first and healing afterward?

## Primary hypotheses

### H1 — Notice-period benefit

Pre-deletion adaptation outperforms post-deletion KD:

\[
RR_{\mathrm{PDRR}} > RR_{\mathrm{hard\ prune+KD}}.
\]

### H2 — Responsibility-transfer benefit

Component-specific missing-function reconstruction outperforms gradual KD:

\[
RR_{\mathrm{PDRR}} > RR_{\mathrm{gradual\ KD}}.
\]

### H3 — Recovery-loss necessity

The full method outperforms a learned Recovery Matrix trained without the component-specific recovery loss.

### H4 — Routing validity

Learned responsibility routing outperforms random routing under the same recovery objective.

### H5 — Causal weighting

Interventional state-position significance improves over otherwise identical non-causal state-position routing.

### H6 — Mechanistic handoff

The post-recovery increase in causal importance of a survivor is positively associated with the responsibility mass assigned to that survivor during training.

### H7 — Scar-free persistence

Any quality advantage remains after:

- the target gate is exactly zero;
- recovery losses and routers are disabled;
- the target layer is physically removed;
- the LoRA/adapter representation, when used, has been merged into ordinary surviving weights;
- the parameter count equals ordinary hard pruning.

## Primary endpoint

The primary behavioral endpoint is the paired difference in the repository's versioned quality score between `causal_state_position_recovery` and `hard_prune_kd`, aggregated over preregistered target layers and seeds.

Secondary comparisons are against `gradual_kd`, `matrix_no_recovery`, `random_matrix_recovery`, and `state_position_no_causal`.

## Target selection

Run exact identity ablations on the intact checkpoint. Rank layers by paired final-logit KL on the frozen calibration split. Select one target from each preregistered quantile:

- low significance: nearest available layer to the 20th percentile;
- medium significance: nearest available layer to the 50th percentile;
- high significance: nearest available layer to the 80th percentile.

Ties are broken by the lower layer index. Layer 0 and the final layer are excluded from the first confirmatory study because they create qualitatively different boundary conditions. Target selection is frozen before recovery outcomes are inspected.

## Seeds and pairing

Confirmatory study: at least three training seeds per target. Every arm uses identical:

- initialization checkpoint;
- target layer;
- training examples and order;
- supervised-token budget;
- optimizer-step budget;
- evaluation examples;
- decoding protocol.

Quality-maximized follow-up runs are reported separately and cannot replace the compute-matched primary comparison.

## Statistical plan

- Preserve per-example results.
- Pair by target layer, training seed, and evaluation example ID.
- Primary uncertainty: cluster bootstrap over target-by-seed cells, with examples retained inside each sampled cluster.
- Primary test: cluster sign-flip/randomization test on target-by-seed mean paired differences.
- Familywise control: Holm correction across the five preregistered behavioral contrasts.
- Report adjusted p-values, 95% confidence intervals, standardized effect sizes, and raw paired differences.
- A contrast must pass both statistical and practical thresholds.

Practical thresholds for the first Rineya study:

- quality-score gain >= 0.005 absolute;
- Recovery-Ratio gain >= 0.10;
- no tracked capability category may regress by more than 0.015 relative to the matched comparator without being explicitly reported as a tradeoff.

These values may be revised only before any confirmatory outcome is inspected, with the revision committed as a new preregistration version.

## Recovery Ratio

For a higher-is-better quality metric:

\[
RR=\frac{Q_{method}-Q_{hard}}{Q_{full}-Q_{hard}}.
\]

For a lower-is-better metric, the implementation first transforms it to utility by negation. If the full-versus-hard denominator is numerically indistinguishable from zero, RR is undefined and must not be reported.

## Handoff Completeness

For divergence `D` from the intact teacher:

\[
H=1-\frac{D(teacher,recovered)}{D(teacher,hard)}.
\]

Values above one are allowed; negative values indicate recovery made the perturbation worse.

## Mechanistic endpoint

For survivor `s`, define causal importance before and after recovery by ablating that survivor from the compact architecture:

\[
I_s^{before}=D(p_{hard},p_{hard,-s}),
\]

\[
I_s^{after}=D(p_{recovered},p_{recovered,-s}),
\]

\[
\Delta I_s=I_s^{after}-I_s^{before}.
\]

The primary mechanistic statistic is the Spearman association between mean route responsibility and `Delta I_s`, evaluated with a recipient-label permutation test. Top-routed/top-gain agreement is secondary.

## Claim ladder

### Claim A — recovery effect

Allowed only if PDRR beats hard-prune KD.

### Claim B — pre-deletion advantage

Allowed only if PDRR also beats gradual KD.

### Claim C — component-specific responsibility transfer

Allowed only if PDRR beats both `matrix_no_recovery` and `random_matrix_recovery`.

### Claim D — causal state-conditioned routing

Allowed only if PDRR beats `state_position_no_causal`.

### Claim E — causal migration of responsibility

Allowed only if the preregistered route-to-causal-gain mechanistic endpoint passes after physical deletion.

No biological-equivalence claim is permitted. Stroke recovery is inspiration, not evidence.

## Falsification criteria

The central responsibility-transfer hypothesis is rejected for the tested setting if any of the following holds:

- gradual KD matches the full method within the equivalence margin;
- random routing matches learned routing;
- removing the recovery loss does not materially change outcomes;
- the advantage disappears in compact-only consolidation;
- the physically pruned model differs materially from the gate-zero model;
- route assignments fail to predict post-recovery causal necessity;
- improvements are explained by unmatched compute, data, or trainable-parameter budgets.
