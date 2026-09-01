# Three-seed replication — findings

**Status: EXPLORATORY_DEV at the preregistered power level. The primary contrast is
complete across three seeds. This is NOT confirmatory: the preregistration in
`PREREGISTRATION.md` is still DRAFT and unsigned, so §8's gates have not formally
been run against a frozen protocol.**

**Branch:** `research/depth-aligned-semantic-transitions-v1` at `31cf2f2`
**Run dates:** 2026-08-31 → 2026-09-01
**Sealed evaluation:** CLOSED, never opened, never read.

Seeds **17, 29, 43**. 4,000 train / 800 DEV problems per run, 250 optimizer steps,
Qwen3-0.6B + LoRA r=16, `checkpoint_readout: final_norm` at layers 7/14/21/27,
`target_centering: within_problem`, greedy generation at `max_new_tokens: 384`,
scorer frozen at `f160f38`. Roughly 50 min per run.

Seed 17's three primary arms are the `g2_budget_*` runs; seeds 29 and 43 are
`g3_s*`. Analysis groups by `(arm, seed)` from each `run_manifest.json`, so the
directory prefix is irrelevant.

## 1. Results

| arm | s17 | s29 | s43 | mean | n |
|---|---:|---:|---:|---:|---:|
| **`cot_baseline`** | 0.8938 | 0.8950 | 0.8925 | **0.8937** | 3 |
| `state_reverse` | 0.8900 | 0.8912 | 0.8950 | 0.8921 | 3 |
| `state_ordered` | 0.8775 | 0.8862 | 0.8862 | **0.8833** | 3 |
| `state_final_repeated` | — | 0.8912 | — | 0.8912 | **1** |

Mechanism metrics, seed-averaged:

| arm | order margin | checkpoint top-1 | monotonic | `L_state` | LM ppl |
|---|---:|---:|---:|---:|---:|
| `cot_baseline` | −0.0011 | 0.2503 | 0.5667 | — | 1.0069 |
| `state_final_repeated` | 0.0071 | 0.2500 | 1.0000 | 0.1433 | 1.0068 |
| **`state_ordered`** | **0.9350** | **0.8332** | **0.9846** | 0.2629 | 1.0076 |
| `state_reverse` | −0.1915 | 0.1033 | 0.0000 | **0.2534** | 1.0075 |

Paired by problem id, clustered bootstrap by `generator` (30 clusters), exact
McNemar on discordant pairs:

| contrast | diff | clustered 95% CI | discordant | p |
|---|---:|---|---:|---:|
| ordered − baseline, s17 | −0.0163 | [−0.0317, −0.0025] | 13:26 | 0.0533 |
| ordered − baseline, s29 | −0.0088 | [−0.0209, +0.0036] | 19:26 | 0.3713 |
| ordered − baseline, s43 | −0.0063 | [−0.0276, +0.0136] | 22:27 | 0.5682 |
| **ordered − baseline, POOLED** | **−0.0104** | — | **54:79** | **0.0370** |
| reverse − baseline, POOLED | −0.0017 | — | 68:72 | 0.8000 |
| ordered − reverse, POOLED | −0.0088 | — | 68:89 | 0.1102 |
| final_repeated − baseline, s29 | −0.0037 | [−0.0186, +0.0122] | 18:21 | 0.7493 |

## 2. The result

> **Depth-aligned semantic states are highly learnable, and aligning them to
> forward reasoning order is the only variant that costs reasoning accuracy.**

The shared projector learns the depth-to-state alignment extremely well — order
margin **0.935**, checkpoint top-1 **0.833** against a 0.25 chance level, monotonic
decoded progress **0.985** — with the language model untouched (perplexity 1.0076
against the baseline's 1.0069). There is no doubt the mechanism trains.

It also does not help. `state_ordered` is **below** plain CoT SFT in all three
seeds, pooled **−1.04 pp** across 2,400 paired problems at **p = 0.037**.

And the cost is specific to the canonical **forward** assignment:

```text
cot_baseline          0.8937  ─┐
state_reverse         0.8921   ├─ mutually indistinguishable
state_final_repeated  0.8912  ─┘   (reverse - baseline: p = 0.80)
state_ordered         0.8833  ←    the only arm that pays (p = 0.037)
```

Reversing the state-to-depth assignment costs **−0.17 pp** (p = 0.80), and at seed
43 it is *positive*. Flattening it to a repeated final state costs −0.37 pp
(p = 0.75, one seed). Only progressive forward ordering measurably hurts.

## 3. No directional advantage — the cleanest single result

`SSOT §5` assigns `state_reverse` the job of testing "directional compatibility of
reasoning and depth". The answer is that there is none detectable, and it is the
result least dependent on the noisy generation endpoint:

| | s17 | s29 | s43 |
|---|---:|---:|---:|
| `L_state`, ordered | 0.2622 | 0.2644 | 0.2621 |
| `L_state`, reverse | **0.2520** | **0.2518** | **0.2563** |

`L_state` is each arm's fit to **its own** targets, so it compares arms on equal
footing. A fully **reversed** assignment is *easier* to fit than the canonical
forward one, in every seed, stable to three decimals. Ordered − reverse on
capability is −0.88 pp pooled (p = 0.11), negative in all three seeds.

The order margin and monotonic columns must **not** be used for this comparison:
both are computed against the canonical order and therefore penalise reverse by
construction. That is why reverse shows margin −0.19 and monotonic 0.0000 while
fitting its own targets better than ordered fits theirs.

Similarly, `state_final_repeated`'s `L_state` of 0.1433 and monotonic score of
exactly 1.0000 are artifacts of its target set: four identical targets are much
easier to fit, and "progress" over four identical states is trivially
non-decreasing. Neither number is comparable across arms.

## 4. Four constraints on reading this

1. **−1.04 pp is BELOW the 2.0 pp practical margin** proposed in
   `PREREGISTRATION.md` §6 *before* these runs. Statistically detectable at this n;
   **not** practically significant by the bar set in advance. That bar is not being
   moved now that the effect landed under it.
2. **Not confirmatory.** The preregistration is unsigned, so this is
   EXPLORATORY_DEV. The §8 gates have not been formally executed against a frozen
   protocol.
3. **G3 has one seed of three.** `state_final_repeated` completed only at seed 29;
   seeds 17 and 43 failed on an environmental accelerator constraint (§5). A
   single-seed null carries ~57% power, so "final-repeated is harmless" is
   suggestive, not established. It is the control that decides whether the ordered
   deficit is about reasoning order or about auxiliary depth losses in general.
4. **Narrow scope.** One 0.6B model, one procedurally-generated corpus, one LoRA
   configuration, one depth schedule. `SSOT §12` lists template overfitting as a
   live threat and no amount of seeds here addresses it.

## 5. Why two runs are missing

Both `state_final_repeated` failures were environmental, not code faults, and the
seven runs that completed did so under lower pressure.

This 8,188 MiB accelerator is shared with the desktop. Measured with **no training
running**, ~20 desktop processes (dwm, explorer, Edge, the Claude app, NVIDIA
Overlay, ShareX, Radeon Software) held **5,592 MiB**, leaving under 2.6 GiB.
Successful runs peaked near 7.8 GiB total. Desktop consumption drifted from roughly
2.5 GiB overnight to 5.6 GiB by 07:00, and every run started after that failed:

| run | failure | at |
|---|---|---|
| `g3_s17_state_final_repeated` | CUDA out of memory | step 1480/2000, then 750/2000 on retry |
| `g3_s43_state_final_repeated` | `CUBLAS_STATUS_INTERNAL_ERROR`, bf16 GEMM | in backward |

The first instance was self-inflicted: the test suite and a summariser validation
were run while training was live, and both touch the accelerator. Recorded in
`9735aed`.

Mitigations added, both protocol-neutral: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
in `scripts/launch_batch.ps1`, and `--min-free-mib` in `scripts/resume_runs.py`,
which waits for headroom rather than starting a run into a squeeze that wastes 50
minutes. Batch sizes and sequence length were deliberately **not** reduced: that
would have changed the config hash and made the re-queued runs a different protocol
from their own seed-mates.

## 6. What this means for the programme

`SSOT §13` lists *"ordered improves mechanism metrics but not capability"* as a
**revise** condition, and that is now measured at the preregistered power level
rather than guessed. Two readings remain, and G3's missing seeds separate them:

- **if final-repeated is confirmed harmless** — the cost is specific to progressive
  forward supervision. That is a sharp, publishable negative result about the tested
  formulation, and it would say the depth axis is indifferent to reasoning order
  while being actively degraded by being told to follow it.
- **if final-repeated also costs ~1 pp at three seeds** — the cost is from carrying
  any consistent auxiliary depth target, the ordered/reverse asymmetry needs
  re-explaining, and the finding becomes about auxiliary-loss interference rather
  than about depth semantics.

Either way the V1 formulation as specified does not improve reasoning, and the
recorded revision candidate remains semantic **motion** rather than absolute state:
`PHASE0_READINESS.md` §3.8 measured between-problem same-stage delta cosine at
0.94–0.99 against within-problem cross-stage −0.32, an exact reversal of the
absolute-state pattern. That direction is **not** implemented and spending the
`SSOT §13` one-mechanism revision budget is a decision, not an implementation
detail.

## 7. Reproducing this table

```bash
python scripts/summarize_runs.py                              # arms x seeds + contrasts
python scripts/summarize_runs.py --baseline-arm state_reverse  # ordered vs reverse
```

Both read completed runs from `runs/`, group by manifest, and name any incomplete
run rather than omitting it silently.

## 8. Claim boundary

Three seeds. 800 DEV problems each. 250 optimizer steps. Unsigned preregistration.
TEST sealed and never read. Nothing here is a Rineya capability claim. The single
contrast reaching p < 0.05 sits below the practical margin fixed in advance, and the
control that would interpret it has one seed of three.
