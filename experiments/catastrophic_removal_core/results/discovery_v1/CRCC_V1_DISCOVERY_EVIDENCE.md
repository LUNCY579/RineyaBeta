# CRCC v1 — first unsealed discovery evidence

**Program status:** **DISCOVERY_COMPLETE__G0_STOP**  
**Preregistration head:** `242613d1b4b5e0a5a4a7154c60337115ca394bd7`  
**Execution commit:** `193f93d8231d25087187a8624d37f86f28dba5bf`  
**Run identity:** `1b91c6d2d8cf68aed50f28e5f1aff838561538eacdcfd3871b3d77f9c7b69896`  
**Stable discovery summary:** `f5ea9f1247eef5ebb4f6d710b0cdddfddcee16f73e4603b811b076f7042884e3`

## Scope and claim boundary

This package contains only the intact Qwen baseline and the 28 preregistered one-layer bypass conditions on deterministic training-corpus diagnostics. It does not contain recovery training, matched-arm design, donor merges, DEV-gate selection, the historical 64-row long-context evaluation, or sealed evaluation.

The historical 64-row evaluator remains outside the canonical registry. Therefore nothing in this package reproduces, replaces, beats, or is directly comparable to Halo's historical `30/64` result.

## Intact baseline

- Diagnostic cases: **32**
- Mean reference NLL: **2.400199**
- Collapse-case rate: **0.0000**
- Intelligibility rate: **1.0000**
- EOS rate: **1.0000**

## Frozen automatic membership

Automatic CRCC layers: **none**.

Manual hypothesis layers remained separate and were not used to alter membership.

## Discovery gate G0

| Check | Result |
|---|---:|
| `all_28_single_layer_conditions_complete` | PASS |
| `intact_baseline_reference_nll_finite` | PASS |
| `automatic_core_depth_in_preregistered_range_1_to_14` | FAIL |
| `manual_hypotheses_not_promoted` | PASS |
| `automatic_members_have_required_cross_source_support` | PASS |
| `diagnostic_case_count_meets_minimum` | PASS |
| `checkpoint_weight_files_hashed` | PASS |
| **G0 overall** | **FAIL** |

## Layer scan

| Layer | Auto CRCC | Score | NLL ratio | Collapse Δ | Intelligibility Δ | EOS Δ | Supporting sources |
|---:|:---:|---:|---:|---:|---:|---:|---|
| 0 | no | 0.4677 | 5.4686 | 0.0000 | 0.0000 | -1.0000 | — |
| 1 | no | 0.5403 | 4.4194 | 0.1562 | -0.2500 | -0.9062 | — |
| 2 | no | 0.5519 | 4.1531 | 0.0938 | -0.5000 | -0.5938 | rag_train, short_preservation_train |
| 3 | no | 0.0411 | 1.2091 | 0.0000 | 0.0000 | 0.0000 | — |
| 4 | no | 0.0728 | 1.4000 | 0.0000 | 0.0000 | 0.0000 | — |
| 5 | no | 0.0529 | 1.2772 | 0.0000 | 0.0000 | 0.0000 | — |
| 6 | no | 0.0552 | 1.2660 | 0.0000 | 0.0000 | -0.0312 | — |
| 7 | no | 0.0368 | 1.1854 | 0.0000 | 0.0000 | 0.0000 | — |
| 8 | no | 0.0223 | 1.0874 | 0.0000 | 0.0000 | -0.0312 | — |
| 9 | no | 0.0405 | 1.2055 | 0.0000 | 0.0000 | 0.0000 | — |
| 10 | no | 0.0265 | 1.1304 | 0.0000 | 0.0000 | 0.0000 | — |
| 11 | no | 0.0260 | 1.1279 | 0.0000 | 0.0000 | 0.0000 | — |
| 12 | no | 0.0804 | 1.4221 | 0.0000 | 0.0000 | -0.0312 | — |
| 13 | no | 0.0233 | 1.1139 | 0.0000 | 0.0000 | 0.0000 | — |
| 14 | no | 0.0000 | 0.9994 | 0.0000 | 0.0000 | 0.0000 | — |
| 15 | no | 0.0203 | 1.0985 | 0.0000 | 0.0000 | 0.0000 | — |
| 16 | no | 0.0282 | 1.1175 | 0.0000 | 0.0000 | -0.0312 | — |
| 17 | no | 0.0099 | 1.0469 | 0.0000 | 0.0000 | 0.0000 | — |
| 18 | no | 0.0440 | 1.2253 | 0.0000 | 0.0000 | 0.0000 | — |
| 19 | no | 0.0340 | 1.1703 | 0.0000 | 0.0000 | 0.0000 | — |
| 20 | no | 0.0055 | 1.0259 | 0.0000 | 0.0000 | 0.0000 | — |
| 21 | no | 0.0058 | 1.0269 | 0.0000 | 0.0000 | 0.0000 | — |
| 22 | no | 0.0313 | 1.1557 | 0.0000 | 0.0000 | 0.0000 | — |
| 23 | no | 0.0106 | 1.0502 | 0.0000 | 0.0000 | 0.0000 | — |
| 24 | no | 0.0161 | 1.0773 | 0.0000 | 0.0000 | 0.0000 | — |
| 25 | no | 0.0056 | 1.0264 | 0.0000 | 0.0000 | 0.0000 | — |
| 26 | no | 0.0615 | 1.3287 | 0.0000 | 0.0000 | 0.0000 | — |
| 27 | no | 0.1730 | 1.4014 | 0.0000 | 0.0000 | -1.0000 | — |

## Provenance

- Official checkpoint: `Qwen/Qwen3-0.6B@c916fa4defd319b7d4e4da17604ca7338f4d99f5`
- Runtime device/precision: `cuda:0` / `bfloat16`
- PyTorch / Transformers: `2.8.0+cu128` / `4.57.1`
- Checkpoint inventory: `48ee98af90eb7815e04e8e4a92b423c5f30963ab7aab5638e9b57dc8e4cf27e1`
- Config SHA-256: `986fc5856dd099d62fac873a566152f83dfdb1a053c430b5daeb5e02ed8770b2`
- Raw workflow artifact: `local-untracked-crcc-v1-local-14572-20260831T224208` in Actions run `local-14572-20260831T224208`
- Sealed evaluation opened: **false**
- Historical 64-row evaluator executed: **false**
- Halo-comparable claim permitted: **false**

## Next permitted action

G0 failed. The experiment must stop for instrument review; no matched recovery arms may be designed from a hand-selected substitute core.
