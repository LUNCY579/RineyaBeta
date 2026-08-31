# Cross-Architecture Graft v1

**Branch:** `research/cross-architecture-graft-v1`  
**Base:** `main@3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`  
**Status:** **BUILT — SYNTHETIC END-TO-END VERIFIED — REAL RINEYA CAMPAIGN UNRUN**

This experiment asks whether a compact Qwen-derived Rineya can acquire a useful
architecture-distinct computation through a small residual sidecar while the
original model remains frozen.

The v1 intervention is

\[
h_{l+1} = F_{\text{host},l}(h_l)
          + g_l(h_l)\,P_{\text{out}}
            F_{\text{foreign}}(P_{\text{in}}h_l),
\]

where the bounded gate is initialized to **exactly zero**. At construction, the
grafted model is therefore the original host, not an approximation to it.

## Claim boundary

The committed evidence proves only that the machinery is internally coherent:

- the zero gate preserves the host output bit-for-bit on the tested path;
- causal donor branches do not read future tokens;
- only graft parameters can be trained;
- frozen host and frozen-donor parameters remain unchanged;
- grouped split and content-hash guards reject data drift;
- a synthetic teacher signal can flow through the full train/export/evidence path;
- matched run manifests can be compared only after model, data, seed, and training
  contracts are proven equal.

It does **not** establish that a cross-architecture graft improves Rineya, that a
latent-KV sidecar is more efficient, or that knowledge has transferred from a
DeepSeek, SmolLM, or Llama checkpoint. Sealed evaluation, the historical 64-row
long-context showcase, and any Halo-comparable claim remain closed.

## What is implemented

### Architecture-neutral graft contract

`GraftBranch` composes:

1. an explicit host-to-foreign representation bridge;
2. an architecture-distinct donor block;
3. an explicit foreign-to-host bridge;
4. a scalar, channel, or token-conditioned bounded residual gate.

`GraftedDecoderLayer` preserves common Hugging Face decoder return contracts and
supports an exact evaluation bypass whenever the gate is still zero.

### Mechanism registry

| Mechanism | Role in v1 | Important boundary |
|---|---|---|
| `latent_kv_attention` | **Primary arm** | Captures shared latent K/V compression; it is not an exact DeepSeek MLA implementation. |
| `nope_attention` | Attention-family matched control and later SmolLM-inspired study | Causal GQA without rotary position transforms; not a transplanted SmolLM checkpoint. |
| `swiglu_mlp` | Ordinary-capacity matched control | Tests whether extra residual capacity alone explains an effect. |
| `routed_moe` | Secondary mechanism scaffold only | Deterministic reference routing; not a complete DeepSeekMoE training recipe. |
| `ForeignModuleAdapter` | Future functional-transfer phase | Wraps an actually loaded donor block behind learned bridges; not used by the primary mechanism-transfer arm. |

The primary experiment deliberately tests **one mechanism at one layer**. The
other implementations do not authorize mixing several architectural changes in
the first campaign.

## Preregistered matched arms

All three plans target zero-indexed layer `14` of the 28-layer Rineya scaffold and
use the same host, data, optimizer, sequence length, steps, and seeds.

| Arm | Foreign width | Trainable graft parameters | Gap versus primary |
|---|---:|---:|---:|
| Latent-KV primary | 512 | 721,409 | — |
| Full NoPE/GQA attention control | 424 | 725,097 | +0.51% |
| SwiGLU ordinary-capacity control | 1,024 | 722,945 | +0.21% |

The attention control is necessary: a win over only an MLP would not show that
latent K/V compression matters relative to generic additional attention.

## Fast verification

From the repository root:

```bash
bash experiments/cross_architecture_graft/run_checks.sh
```

Windows PowerShell:

```powershell
./experiments/cross_architecture_graft/run_checks.ps1
```

The check suite compiles the package, runs all unit/integration tests, validates
all three plan configurations, executes the deterministic CPU smoke run in a
temporary directory, and verifies the resulting evidence payload hash.

Individual commands:

```bash
python -m experiments.cross_architecture_graft.src.runner \
  --config experiments/cross_architecture_graft/configs/latent_kv_primary_plan.json \
  --command preflight

python -m experiments.cross_architecture_graft.src.runner \
  --config experiments/cross_architecture_graft/configs/smoke.json \
  --command smoke \
  --out /tmp/cag-smoke.json

python -m experiments.cross_architecture_graft.src.verify_evidence \
  /tmp/cag-smoke.json
```

## Real-campaign path

The real runner is intentionally local-only. It will not download a model, use a
remote dataset, invoke a sealed evaluator, or silently create a row-random split.

1. Inspect the intended unsealed corpus for complete generator/template provenance:

   ```bash
   python -m experiments.cross_architecture_graft.src.inspect_data \
     --input path/to/open_cag_mix.jsonl \
     --out artifacts/cross_architecture_graft/data/inspection.json
   ```

2. Freeze a grouped train/dev split. A row without an explicit provenance key is
   rejected:

   ```bash
   python -m experiments.cross_architecture_graft.src.prepare_split \
     --input path/to/open_cag_mix.jsonl \
     --train-out artifacts/cross_architecture_graft/data/unsealed_train.jsonl \
     --dev-out artifacts/cross_architecture_graft/data/unsealed_dev.jsonl \
     --manifest-out artifacts/cross_architecture_graft/data/unsealed_split_manifest.json \
     --group-fields template_id,generator_id,group_id,cag_group_id,source_family
   ```

3. Materialize the three arms across the five frozen seeds:

   ```bash
   python -m experiments.cross_architecture_graft.src.materialize_campaign \
     --model-path artifacts/rineya-core-en32k-28L \
     --train-mix artifacts/cross_architecture_graft/data/unsealed_train.jsonl \
     --dev-mix artifacts/cross_architecture_graft/data/unsealed_dev.jsonl \
     --split-manifest artifacts/cross_architecture_graft/data/unsealed_split_manifest.json \
     --output-dir artifacts/cross_architecture_graft/campaign_v1
   ```

4. Run one materialized config at a time:

   ```bash
   python -m experiments.cross_architecture_graft.src.runner \
     --config artifacts/cross_architecture_graft/campaign_v1/configs/20260831/latent_kv.json \
     --command real
   ```

5. Compare same-seed unsealed NLL health screens only after comparability proof:

   ```bash
   python -m experiments.cross_architecture_graft.src.compare_manifests \
     artifacts/cross_architecture_graft/campaign_v1/runs/20260831/latent_kv/evidence_manifest.json \
     artifacts/cross_architecture_graft/campaign_v1/runs/20260831/nope_gqa/evidence_manifest.json \
     artifacts/cross_architecture_graft/campaign_v1/runs/20260831/swiglu/evidence_manifest.json \
     --reference-id cag-v1-latent_kv-seed-20260831
   ```

NLL is only a pipeline and optimization-health screen. Promotion requires the
paired open-development semantic gates and seed-level criteria defined in
`EXPERIMENT_PROTOCOL.md`.

## Directory map

```text
cross_architecture_graft/
├── configs/                  strict smoke, primary, and matched-control plans
├── results/                  committed evidence descriptions / reference manifests
├── src/
│   ├── bridges.py            explicit representation maps
│   ├── donors.py             architecture-distinct mechanisms
│   ├── graft.py              exact-zero gated wrapper
│   ├── integration.py        layer discovery, attachment, graft-only packs
│   ├── config.py             strict schema and closed-evaluation guards
│   ├── splits.py             deterministic grouped split
│   ├── inspect_data.py       provenance-field audit
│   ├── materialize_campaign.py matched five-seed config generation
│   ├── real_runner.py        local unsealed train/dev path
│   ├── compare_manifests.py  strict same-seed comparability check
│   └── evidence.py           content-addressed evidence manifests
├── tests/                    deterministic unit and end-to-end tests
├── EXPERIMENT_PROTOCOL.md    preregistered scientific contract
├── RUNBOOK.md                operational sequence and failure handling
└── BUILD_NOTES.md            implementation decisions and current evidence
```

## Scientific interpretation

There are two distinct future questions:

1. **Mechanism transfer:** does an architecture-distinct, newly initialized
   sidecar learn more useful updates than equally sized conventional capacity?
2. **Functional transfer:** can a frozen block from another trained model family
   be mapped into Rineya through learned input/output stitches and outperform
   reinitialized, shuffled, and same-family donor controls?

V1 answers neither yet. It creates a falsifiable and recoverable path to test the
first question without conflating it with the second.

## References that motivate the design

- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437 — MLA, MoE, and
  multi-token-prediction provenance.
- Hugging Face, *SmolLM3: smol, multilingual, long-context reasoner* — periodic
  NoPE/GQA design provenance.
- Hernandez et al., *Model Stitching: Looking For Functional Similarity Between
  Representations*, arXiv:2303.11277 — learned interfaces between differently
  shaped representations.
- Chen, Goodfellow, and Shlens, *Net2Net: Accelerating Learning via Knowledge
  Transfer*, arXiv:1511.05641 — function-preserving architecture transformation
  as a governing safety principle.
