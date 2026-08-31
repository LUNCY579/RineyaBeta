# DAST progress

## 2026-08-31 — post-review corrections to the V1.1 commit

An adversarial review of `30119fe` over four lenses (correctness,
scientific-validity, reproducibility, test-quality) raised 29 findings, each
verified or refuted by an independent agent. It caught **a real regression
introduced by that commit**, plus several repairs whose tests did not actually pin
them. Recorded here rather than amended into `30119fe`, because "a review caught
this" is part of the provenance.

### The regression

Adding `time` and `statistic` to the scorer's numeric-tolerance set routed them
through `_numeric_value`, which uses `re.search` and therefore compared only the
**first** number in each answer. Verified by execution:

```text
answers_equal("16:20", "16:55", "time")                            -> True   WRONG
answers_equal("mean 3.5, range 12", "mean 3.5, range 99", ...)     -> True   WRONG
```

All 160 DEV `time` answers are `HH:MM`, so the minute field was always discarded;
24 of 80 `statistic` answers are `mean X, range Y`, so the range was always
discarded. In the shipped 800-problem eval window that is ~61 problems (7.6%).
Both were **False** before `30119fe` under exact string equality, so the commit
made the primary endpoint worse for those types: it inflates absolute accuracy and
converts genuinely discordant pairs into ties, which is exactly the paired power
the endpoint exists to provide.

Fixed by comparing **all** numbers elementwise when the gold carries more than
one, while keeping first-number comparison for single-number golds so the intended
unit tolerance (`129 N` vs `129`, `$14.50` vs `14.50`) is preserved. The reviewer's
`rate` claim did **not** hold and was not acted on: all 80 DEV `rate` answers are
single-number-plus-unit.

### A no-op hiding inside the readout repair

Chasing "these tests are vacuous" produced a more important finding. The shared
projector begins with its own `LayerNorm`, which absorbs any **uniform** affine
transform exactly. So `checkpoint_readout: final_norm` does nothing at all when the
final norm's gain is uniform — and the toy backbone's `final_norm` was at default
init (weight 1, bias 0), on top of `ToyBlock` already ending in a `LayerNorm`.

Consequence: the offline smoke and every unit test would have passed with the
readout entirely broken. The toy now carries a deterministic **per-dimension**
gain, which is what a trained model has and the only reason the readout measurably
de-correlated the checkpoints on Qwen3-0.6B (7v21: 0.954 → 0.689).

### Other confirmed defects, now fixed

| Defect | Fix |
|---|---|
| The `final_norm` guard tested the configured **fraction**, but `int(round(L·f))` maps all of `[1 − 0.5/L, 1.0)` onto layer `L`. `0.99` on an 8-layer model, or the shipped `0.964` on any model with `L ≤ 13`, resolved to the top layer while passing validation | assert on the **resolved depth** in `DASTModel.__init__`, where `checkpoint_layers` and `num_hidden_layers` are both known |
| `_run_status` took its scope from `eval_split` alone, so a run that **trained** on the sealed split reported `..__DEV__..` | consider both splits |
| `_encode_targets` defaulted `reference_texts=None`, silently centering on the arm's own grid — the mode its own docstring forbids, which would unmatch the controls and collapse `state_final_repeated` to a zero target | refuse without an explicit canonical reference |
| The three new `Literal` config fields were unvalidated, so a typo survived config load and surfaced as a bare `KeyError` after the model and corpus were in memory | validated by value; `confirmatory_dev` additionally requires `preregistration_frozen` |
| No arm config set `evidence_status`, so all six DEV arm runs would have emitted the same manifest status as the 16-problem rehearsal | `base.yaml` sets `exploratory_dev`; the rehearsal overrides to `engineering` |
| The smoke config pinned the **pre-repair** geometry, so the only end-to-end offline test never exercised either protocol correction | smoke now runs `final_norm` + `within_problem` at fractions ending 0.9 |

### Repairs that had no test, and now do

Left-padded generation, seed-before-construction, the feedback hook readout, the
shipped-init intervention no-op, and the smoke device fix were all revertible with
the suite still green. `tests/test_v11_review_fixes.py` pins each. Two assertions
in the first pass were tautological (`allclose(last, last)`) and are replaced with
the three facts actually verified on Qwen3-0.6B: `lm_head(hidden_states[-1])`
reproduces `logits`, `lm_head(hidden_states[-2])` does not, and the last entry must
never be normed twice.

Suite is now **94 tests**, passing with a GPU visible, with `CUDA_VISIBLE_DEVICES=-1`,
and under CI-like invocation (`PYTHONPATH=src` only). Smoke PASS, and the real
seam rehearsal still exits 0 with `checkpoints [7, 14, 21, 27]`,
`readout final_norm`, `centering within_problem`.

### Not acted on

One reviewer claim is **declined**: that per-split corpus files make
`validate_dataset`'s cross-split duplicate detection unable to fire. True as
stated, but the corpus itself asserts `cross_split_problem_overlap: false` in
`dataset_stats.json` and ships `source_pair_sha256` per record. The right response
is a cross-file contamination check as a separate tool, not a change to this
commit. Recorded in `docs/PHASE0_DEFECT_AUDIT.md`.

---

## 2026-08-31 — V1.1 readiness correction

**Status: BUILT — UNRUN on a qualifying arm run. TEST CLOSED.
G0 now passes on the GPU host. The real Qwen/LoRA seam executes end to end.**

The scientific status is unchanged: no arm has been trained, no DEV comparison
exists, and nothing here supports a Rineya capability claim. What changed is that
the real-model path now runs at all, and the Phase-0 diagnostics the SSOT asks for
have been measured on the real checkpoint.

Full measurements: `docs/PHASE0_READINESS.md`.
Triaged defect audit (mostly UNVERIFIED, read its header): `docs/PHASE0_DEFECT_AUDIT.md`.

### Why the previous entry's qualification record was misleading

The 2026-08-30 entry recorded `pytest -q PASS (37 tests)` and `smoke PASS` without
naming a host. Measured on this GPU host, same commit, same interpreter, varying
only device visibility:

```text
                                  GPU visible        CUDA_VISIBLE_DEVICES=-1
python -m compileall -q src tests  PASS               PASS
python -m ruff check src tests     FAIL (1 error)     FAIL (1 error)
pytest -q                          36 passed 1 FAIL   37 passed
python -m dast.cli smoke           FAIL               PASS
```

So the recorded PASS was reproducible only on a CPU-only host and failed on the
host that has to run the science. Qualification records now name the host.

### Blockers repaired

| # | Defect | Repair |
|---|---|---|
| B1 | `apply_chat_template(tokenize=True)` returns a mapping on transformers 5.x, so `_as_int_list` raised `ValueError: invalid literal for int(): 'input_ids'` and **every** real-model path died at tokenization | unwrap `input_ids` when the return value is a mapping |
| B2 | `smoke.py` compared a CPU snapshot against CUDA parameters, so the offline qualification crashed on any CUDA host | compare on a common device |
| B3 | one ruff error gated the CI Lint step red, so compile/pytest/smoke never ran in CI at all | fixed |
| B4 | `data.path` pointed at a metadata-only manifest (no `problem`, `answer`, `full_response`, `reasoning_steps`, `compact_state_trace`); `validate-data` failed on all 48,000 records, and train+eval were derived from one path while the corpus ships one file per split | optional `data.train_path` / `data.eval_path`, pointed at `data/full_context/{train,dev}.jsonl` |
| B5 | generation right-padded and decoded from the batch-max width | left-pad, decode per row, and build the mask from known pad counts rather than by value comparison |
| B6 | the endpoint measured the harness: 51% of generations hit the 128-token cap without EOS, and `answers_equal` gave numeric tolerance to only 30.0% of DEV answer types | cap 384 (hit-cap 0.010), and answer-type-aware comparison for measurement/decimal/money/time/set/ordering/boolean |
| B7 | `command_inspect_alignment` read records with no `assert_split_allowed`, so `--split test` printed compact states — including `"Final answer: ..."` — with `allow_sealed_test: false` | guarded; verified to raise |
| B8 | `set_global_seed` ran inside `train()`, i.e. **after** the LoRA adapter and shared projector were constructed, so the same config and seed produced different initial weights on every run | seed first in `command_train` |

### Phase-0 protocol corrections, selected in the shipped configs

Both are switchable, both default to the previously committed behaviour, and both
are recorded in the run manifest so a run is not interpretable without them.

- `model.checkpoint_readout: final_norm` with `checkpoint_fractions` ending at
  **0.964 → layer 27, not 28**. `hidden_states[28]` is post-final-RMSNorm
  (`lm_head(h[28])` reproduces `logits` to 0.0 absolute error) while 7/14/21 are
  raw residual states (elementwise RMS 49/49/49.5 versus 3.06), so a schedule
  pinned to 1.0 fed two representation classes into the single shared projector.
  Measured pooled cosine 7v28: −0.018 raw, −0.111 after norming 7/14/21 only,
  **+0.409** for the 7/14/21/27 schedule — the only variant with no
  near-orthogonal outlier. Config validation now rejects `final_norm` combined
  with a final fraction of exactly 1.0.
- `semantic.target_centering: within_problem`. As shipped, targets carried 56.6%
  of their variance in the problem factor and 9.3% in the stage factor, while
  hidden states carried 90.8% stage and 6.0% problem — the two sides of the
  objective encoded orthogonal factors. Consequence: `state_ordered` sat at
  **0.860** cosine from `state_shuffled`, so the arms shared ~86% of their target
  signal. Centering each problem on its own canonical-state mean moves that to
  **−0.019** and lifts the frozen shared-linear-probe order margin from **0.129 to
  0.977** (0.995 with the readout fix). The mean is taken over canonical states,
  never the arm's own targets, so it is identical across arms and
  `state_final_repeated` still receives a constant non-zero target.

`training.batch_size` 1 → 2 with accumulation 16 → 8. The effective batch and the
optimizer-step budget are unchanged, so arms stay matched, but
`multi_positive_contrastive_loss` returns exactly 0.0 when `batch < 2`, so the
documented anti-collapse term had been contributing nothing. The rehearsal below
reports `loss/contrastive: 0.855`, non-zero for the first time.

### Honesty repairs, no behaviour change

- The smoke previously overwrote the feedback decoder (`normal_(std=0.02)`) and
  the gates (`fill_(2.0)`) and reported only the perturbed logit delta. Because
  `feedback_decoder.weight` is zero-initialised by contract, `delta = decoder(state)
  * gate` is identically zero, so **every intervention is a bitwise no-op until the
  decoder has trained**. The smoke now reports both numbers:
  `shipped_init_max_logit_delta: 0.0`, `shipped_init_is_no_op: true`, alongside the
  perturbed delta that qualifies the path. An untrained feedback run cannot produce
  a causal result and must not be reported as one.
- The code comment claiming `tanh(-3)` "starts near zero" was wrong: it is −0.995,
  near saturation, which also attenuates the gate's own gradient ~100×.
- The toy backbone appended only raw post-layer residuals, so `hidden_states[-1]`
  was pre-norm while Hugging Face's is post-norm. The offline suite and CI were
  therefore structurally blind to the exact asymmetry that only appears on a real
  checkpoint. The toy now matches HF semantics.
- The feedback hook pooled the pre-norm layer output while the auxiliary path read
  post-norm `hidden_states[...]`, so a projector trained under the auxiliary config
  was silently fed a different tensor during the causal probe. Both paths now apply
  the same readout.
- Manifest status was derived from `eval_split` alone, so a 16-problem seam
  rehearsal was recorded as `DEV_COMPLETE`, indistinguishable from a qualifying arm
  run. `experiment.evidence_status` now defaults to `engineering` and the status is
  composed as `TIER__SCOPE__PHASE`; the sealed-split test is case-insensitive.
- `run_manifest` schema v2 hashes **every** corpus file a run touched, instead of
  one path.

### Qualification, on this host

```text
host        RTX 4070 Laptop 8188 MiB, driver 610.62
python      3.11.15   torch 2.13.0+cu130   transformers 5.15.0   peft 0.20.0
base model  Qwen/Qwen3-0.6B, HF cache revision c1899de289a04d12100db370d81485cdf75e47ca

python -m compileall -q src tests                       PASS
python -m ruff check src tests                          PASS
pytest -q                                               PASS (74 tests, GPU visible AND CPU-only)
python -m dast.cli smoke                                PASS (GPU visible)
dast validate-data .../full_context/dev.jsonl           PASS (2400 records, 0 errors)
dast inspect-alignment --split test                     PermissionError, as required
dast inspect-alignment ordered/shuffled/reverse         manual RUNBOOK section 3 checks pass
dast train --config configs/seam_rehearsal.yaml         EXIT 0
```

The RUNBOOK §3 checks were confirmed on real records for the first time: the
initial `"No derived facts yet."` state is excluded, ordered targets progress
forward, shuffled is non-identity with the same multiset, reverse is exact, and
variable-length traces record repeated source indices (canonical `[0,1,3,4]` from
five targets).

### Seam rehearsal — ENGINEERING ONLY

`configs/seam_rehearsal.yaml`, 16 train and 8 eval problems, generation disabled.
This exists to prove the seam executes and to measure cost. It is far too small to
support any scientific claim and must never be aggregated with arm runs.

```text
status                    ENGINEERING__DEV__COMPLETE
resolved checkpoints      7, 14, 21, 27
readout / centering       final_norm / within_problem
precision                 bf16
LoRA trainable params     10,092,544 of 606,142,464 (1.69%)
optimizer steps           1        global steps 8
loss/contrastive          0.855    (was identically 0.0 at batch_size 1)
loss/state                0.963
state/order_margin        0.040    at random projector init, 1 optimizer step
```

The order margin here is **not** comparable to the 0.977 probe figure: that was a
least-squares probe *fitted* to held-out data, this is an untrained MLP projector
after one step. The comparison to make later is Δmargin over the measured frozen
floor, and ordered's Δ against shuffled's and reverse's — not any absolute margin.

### Measured cost, for budgeting Phase 1

| | s/problem | 800 DEV | × 18 runs |
|---|---:|---:|---:|
| batch 4, cap 128 | 1.41 | 19 min | 5.6 h |
| batch 4, cap 384 | 2.78 | 37 min | 11.1 h |

Inference is ~2.5 GiB resident, so 8 GiB is not the binding constraint; decode
throughput is.

### Next permitted action

**Not** the Phase-1 matrix. In order:

1. Re-measure the Phase-0 floors *under the corrected readout and centering*, at
   n ≥ 2,000. Every number in `docs/PHASE0_READINESS.md` was measured under the
   old geometry, and the corrected geometry pushes the untrained order margin near
   saturation — which makes mechanism metrics necessary but non-discriminative.
2. Restate G1 as a delta over that measured floor, and as
   Δmargin(ordered) > Δmargin(shuffled) and > Δmargin(reverse). "Margin is
   positive" is satisfied by the frozen pretrained model and proves nothing.
3. Verify the `docs/PHASE0_DEFECT_AUDIT.md` backlog. 55 of its adversarial
   verifiers died on a session limit, so apart from the rows marked CONFIRMED it
   is a triage queue, not evidence.
4. Only then resolve and freeze `docs/PREREGISTRATION.md`, and record the Stage-C
   design decision.

Operators and feedback remain phase-gated and must not begin because the software
runs.

### Not done, and why

- The Phase-1 matrix, the preregistration freeze, and the Stage-C decision are the
  operator's calls, not consequences of these repairs.
- A stage-contrastive term, a hidden-state semantic-motion head
  (`P_Δ(H_{k+1} − H_k) → O_k`), and any depth-schedule search are deliberately
  **not** implemented. Each is a new mechanism, and SSOT §13 allows exactly one
  bounded revision; spending it is a recorded decision, not an implementation
  detail. The motion direction is supported by measurement (between-problem
  same-stage delta cosine 0.94–0.99 versus within-problem cross-stage −0.32) but
  scored **worse** than the corrected state route on held-out data (margin 0.819
  versus 0.995), so it belongs in Phase 2, not V1.
- Restricting Phase 1 to traces with ≥K distinct states was measured and
  **rejected**: it moves the ordered/shuffled contrast only 0.860 → 0.849 while
  discarding 43.8% of the corpus.
- The repository-wide `docs/EXPERIMENT_PROGRAM_AUDIT.md` still does not mention
  DAST and still asserts the remote exposes one live branch. That is a separate
  documentation commit.
