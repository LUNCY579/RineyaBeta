# PDRR controlled CPU reference study

> This is a residual-sequence mechanism study, not a Rineya checkpoint benchmark. It cannot support a claim that PDRR improves an LLM.

Mode: **full preregistered reference**

| Arm | Quality ↑ | Recovery Ratio ↑ | Handoff ↑ | Route/gain ρ ↑ |
|---|---:|---:|---:|---:|
| `causal_state_position_recovery` | 0.795245 | 0.319 | 0.368 | 0.167 |
| `full` | 1.000000 | 1.000 | 1.000 | nan |
| `gradual_feature_kd` | 0.767822 | 0.228 | 0.056 | nan |
| `gradual_kd` | 0.743503 | 0.147 | -0.281 | nan |
| `hard_prune` | 0.699329 | 0.000 | 0.000 | nan |
| `hard_prune_kd` | 0.763044 | 0.212 | -0.097 | nan |
| `hard_prune_sft` | 0.689898 | -0.032 | -1.134 | nan |
| `matrix_no_recovery` | 0.743503 | 0.147 | -0.281 | 0.167 |
| `random_matrix_recovery` | 0.772763 | 0.243 | 0.166 | -0.167 |
| `state_matrix_recovery` | 0.778459 | 0.263 | 0.244 | 0.000 |
| `state_position_no_causal` | 0.779671 | 0.267 | 0.245 | 0.500 |
| `static_matrix_recovery` | 0.779789 | 0.267 | 0.108 | -0.333 |
| `uniform_matrix_recovery` | 0.779789 | 0.267 | 0.108 | -0.333 |

## Validation

- PASS — `expected_arm_seed_cells`
- PASS — `all_registered_arms_present`
- PASS — `hard_pruning_causes_quality_damage`
- PASS — `all_gate_zero_physical_checks_pass`
- PASS — `all_compact_parameter_frontiers_match_hard_pruning`
- PASS — `full_method_beats_random_router_in_reference`
- PASS — `full_method_beats_gradual_kd_in_reference`

## Confirmatory claim gates

- NOT SUPPORTED — `recovery_effect`
- NOT SUPPORTED — `predeletion_advantage`
- NOT SUPPORTED — `component_specific_transfer`
- NOT SUPPORTED — `causal_routing`
- NOT SUPPORTED — `mechanistic_migration`

Claim gates may remain unsupported in this small study because three paired seeds cannot provide strong confirmatory power. That is an intended safeguard, not an error.
