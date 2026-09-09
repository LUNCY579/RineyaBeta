# PROPOSED amendment to the HLGA-1 P2 equivalence criterion

**STATUS: PROPOSAL. NOT AUTHORIZED. NOT EXECUTED.**
Requires human review and sign-off before any run. Nothing in this document
licenses P3, P4 or P5, and no threshold here has been applied to any receipt.
Written *before* seeing any P3 outcome, so that it cannot be tuned to one.

## Why an amendment is needed

P2 fails on the real R2 checkpoint in all four registered device/backend
combinations, on `local_stream`, `all_local_stream` and `frontloaded_stream`,
by roughly 1e-4 against `atol = rtol = 2e-05`. The mechanism is not at fault:

* the attended absolute key positions are identical between the full and
  chunked schedules in every arm, 0 of 28 layers diverging, no tolerance used;
* `argmax_identical` holds in every arm and every combination;
* `native_global` and `gate_identity` are bit-exact at `0.0` on both math
  backends.

The residual is floating-point non-associativity. Chunking the query dimension
changes the gather width, so the same softmax accumulates over differently
shaped matmuls; at layer 0 of the real model, with the softmax following the
input dtype, full-vs-stream differs by 11.8 ulps in float32 and 15.2 ulps in
float64, and over 28 residual layers this compounds to ~1e-4 in the logits.

The instrument, not the retrofit, is the problem, on three counts:

1. **It is unachievable.** No correct implementation of chunked prefill can
   meet an absolute 2e-05 logit bound at 28 layers in float32.
2. **It is not discriminating.** `sink_stream` clears the bar at 3.624e-05 on
   cuda/math while `local_stream` fails at 1.106e-04; the two arms differ only
   by four sink slots. On cuda/sdpa even `native_global`, the independent
   parity oracle, fails at 2.766e-05. Pass and fail track rounding luck.
3. **It was never exercised.** Every receipt committed to this branch before
   this work carries `real_rineya_checkpoint_loaded: false`. The threshold was
   calibrated on the 4-layer, hidden-32 CPU fixture, where accumulation depth
   is roughly thirty times smaller.

The registered threshold was **not** changed. The failing receipts stand.

## Proposed replacement criterion

Replace the single absolute `allclose` with a conjunction of exact conditions
plus one control-derived bound. Conditions 1-4 are exact and carry no tolerance
at all; only condition 5 admits a number, and that number is *measured on a
known-correct arm in the same run* rather than fixed in advance.

A P2 check **passes** only if all of:

1. **Attended-position identity (exact).** For every layer, the set of absolute
   key positions each query attends to is identical between the full forward
   and the chunked streaming forward. Already implemented and already passing.
2. **Argmax identity (exact).** Greedy token choice identical at every
   position. Already implemented and already passing.
3. **Native parity (exact) on the math backend.** `native_global` must be
   bit-exact `0.0` against the native HF eager forward. Already passing on
   cpu/math and cuda/math. On `sdpa` the kernel is a different implementation,
   so this condition applies to the math backend only and `sdpa` is qualified
   against the math backend's research forward instead.
4. **Gate identity (exact).** `A0 == A4`, `A2 == A5`, `A3 == A6` bit-exact
   before training. Already passing at `0.0`.
5. **Drift within the all-global control.** For each windowed arm, the
   full-vs-stream `max_abs_error` must not exceed `K` times the SAME quantity
   measured on `A0_global` in the same run, same device, same backend, same
   seed. `A0_global` is known-correct by condition 3, so its drift is the
   arithmetic noise floor for that configuration. `K` is registered before any P3 run; see the
   calibration below. On the four existing receipts the observed ratios are:

   | arm | cpu/math | cpu/sdpa | cuda/math | cuda/sdpa |
   |---|---|---|---|---|
   | local_stream | 4.67 | 3.39 | 3.05 | 2.06 |
   | all_local_stream | 4.59 | 4.83 | 5.03 | 3.08 |
   | frontloaded_stream | 6.10 | 3.15 | 2.08 | 3.55 |

   Maximum observed 6.10 at the registered qualification configuration. A
   correct implementation at a different configuration (200 tokens, window 16)
   reaches 7.3, so the bound must not be pinned to the edge of what has been
   observed. **Proposed `K = 50`**, chosen from the measured gap below rather
   than from the correct side alone.

### Calibrating `K` against real faults, not against the correct run

Three plausible eviction bugs were injected into the real 28-layer checkpoint
(200 tokens, window 16, four prefill chunks plus decode) and measured against
the same `A0_global` noise floor:

| arm | injected fault | max_abs_error | ratio to A0 | argmax identical | positions identical |
|---|---|---:|---:|---|---|
| A2 | *none* | 1.576e-04 | 7.3 | yes | **yes** |
| A2 | retain two slots short | 3.68 | 170,040 | no | no |
| A2 | positions renumbered by 1 | 7.89 | 364,635 | no | no |
| A3 | *none* | 2.098e-05 | 1.0 | yes | **yes** |
| A3 | retain two slots short | 1.69 | 78,162 | **yes** | no |
| A3 | sinks dropped on eviction | 27.2 | 1,258,638 | no | no |
| A3 | positions renumbered by 1 | — | — | merge's contiguity guard raised | |

Three things follow, and they are the reason the criterion is shaped this way:

* **Correct and faulty are separated by four orders of magnitude** (7.3 versus
  78,162 at the closest approach). `K = 50` sits ~7x above the correct side and
  ~1,500x below the nearest fault, so it is not finely tuned in either
  direction. A signer should still confirm it independently.
* **Argmax identity alone is NOT sufficient.** `retain two slots short` on A3
  moves the logits by 1.69 absolute -- 78,000x the noise floor -- and argmax
  stays identical. A criterion resting on argmax would have passed a genuinely
  broken cache.
* **Attended-position identity caught every injected fault**, exactly and with
  no tolerance. Condition 1 is the real detector; condition 5 is a backstop for
  faults that somehow preserve the position sets. This is now a standing test
  (`tests/test_boundaries.py::InstrumentPowerTests`), so the check cannot
  silently lose its power to a later refactor.

Rationale for the shape of condition 5: it asks the question P2 actually cares
about -- *does windowing introduce error beyond what chunking alone already
introduces?* -- rather than the question the current threshold asks, which is
*is 28-layer float32 accumulation smaller than an arbitrary constant?*

## What must NOT change with it

* No change to arms, topologies, window sizes, sink counts or global placement.
* No change to the primary contrast `A3_sink512` vs `A0_global`.
* No change to dtype, backend set, block size, batch size or decoding.
* No relaxation of any exact condition. Conditions 1-4 stay pass-or-fail.
* The amendment must be committed and signed off **before** the P2 rerun that
  uses it, and the resulting receipt must record which criterion version it
  was judged under.

## Second amendment, required in the same pass: the DEV set cannot move

Computed from the case generator and the window rule alone, with no model
forward (see `AGENT_REAL_MODEL_QUALIFICATION.md`):

* At W=512, the answer tokens sit outside the local window in **12 of 72**
  cases when the question is read, and **24 of 72** by the end of a 24-token
  generation. The other cases cannot show window damage.
* At W=1024, **0 of 72** in either case. `A1_local1024` and `A7_sink1024`
  therefore cannot exhibit any window effect at all on the registered lengths,
  which makes the `window_budget` and `sink_at_1024` contrasts uninformative
  as registered.
* The four prefix sinks never hold evidence. The earliest evidence token
  across all 72 cases is at position 21, while the sinks cover positions 0-3,
  which decode to `'Read the records.'`. The `sink_at_equal_budget` contrast
  therefore tests retention of the *instruction prefix* and attention-sink
  stabilisation, not evidence retention. That is a legitimate question, but it
  is not the question the contrast name suggests.

Consequence: the overall 72-case McNemar is diluted roughly 3:1 by cells where
the mechanism cannot bite, and with 12-24 informative pairs it has no
meaningful power (six discordant pairs all one way gives p = 0.031, five gives
0.063). Reading the overall number as "no damage" would be a power artefact.

Minimum viable fix, to be reviewed alongside the tolerance change: add prompt
lengths beyond 1025 -- at least one stratum near 2048 -- so that a 1024-wide
window can evict, and report the `by_length` x `by_depth` cells at the long
lengths as the primary read rather than the pooled 72. Any change to the case
set changes `diagnostic_id`, so it must be registered before P3 is run, not
after.

## What would still be true after both amendments

P3 would remain a bounded, correlated, exploratory open-DEV diagnostic. It
would still license no claim about quality, preservation, speed or memory, and
P4/P5 would remain closed. The amendments make P2 able to fail for the right
reason and P3 able to move at all; they do not upgrade what either can claim.
