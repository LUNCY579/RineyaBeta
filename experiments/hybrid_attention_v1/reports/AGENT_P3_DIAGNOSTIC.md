# HLGA-1 P3: the open synthetic DEV diagnostic ran, and its capability endpoint is readout-limited

Run `hlga-p3-20260909-cuda-math`, cuda/math, block 128, cap 6144 MiB,
72 cases x 10 arms = 720 generations, greedy, max_new_tokens 24,
`eos_token_ids = [32742, 32744]`,
`diagnostic_id = 7fce6dc35b9ca06de1114b34b0f475ebc193047878be66813683b7740b7e1d65`.
Authorized by P2 receipt `hlga-p2-gate-20260909-cuda-math`
(`receipt_sha256 04c9bd54...`, code identity `0718c7a1c019...`), which the
diagnostic receipt names explicitly.

**Headline: the registered capability endpoint is at the floor for every arm,
so the registered primary contrast is uninformative. It is NOT evidence that
the retrofit preserves capability.** Two other results do stand, and one of
them is strong.

---

## 1. The registered endpoint floored

Every arm scored `trimmed_exact = 0.000`, `strict_exact = 0.000`,
`answerable_accuracy = 0.000`, with `budget_exhaustion_rate = 1.000` and
`stopped_eos_rate = 0.000`. Consequently **all ten registered contrasts are
0 wins / 0 losses / delta +0.000 / McNemar p = 1.000**, including the primary:

| contrast | reference -> treatment | wins | losses | delta | p |
|---|---|---:|---:|---:|---:|
| **primary_frozen_sink_retrofit** | **A0_global -> A3_sink512** | **0** | **0** | **+0.000** | **1.000** |
| pure_window_damage | A0_global -> A2_local512 | 0 | 0 | +0.000 | 1.000 |
| sink_at_equal_budget | A2_local512 -> A3_sink512 | 0 | 0 | +0.000 | 1.000 |
| window_budget | A1_local1024 -> A2_local512 | 0 | 0 | +0.000 | 1.000 |
| sink_at_1024 | A1_local1024 -> A7_sink1024 | 0 | 0 | +0.000 | 1.000 |
| periodic_global_access | A8_all_local512 -> A2_local512 | 0 | 0 | +0.000 | 1.000 |
| placement_at_equal_work | A9_frontloaded512 -> A2_local512 | 0 | 0 | +0.000 | 1.000 |
| the three gate-identity contrasts | — | 0 | 0 | +0.000 | 1.000 |

A tie at the floor is **readout-limited, not noninferiority**, exactly as
`COMPATIBILITY_CONTRACT.md` warned. `A3 ~= A0` here means "neither arm could
be scored", not "the retrofit is harmless".

### Why it floored

The base model answers, then never stops. On the reference arm A0_global, an
instrument diagnosis (**reference arm only, deliberately not computed per arm
so it cannot preview any contrast**):

| | A0_global |
|---|---:|
| registered `trimmed_exact` (full-string equality) | **0 / 72** |
| correct answer emitted as the FIRST generated content | **47 / 72** |
| — of the 36 answerable cases | 22 / 36 |
| — of the 36 unanswerable cases, correctly began `UNKNOWN` | 25 / 36 |
| `stopped_eos` | 0 / 72 |
| `budget_exhausted` | 72 / 72 |

Concretely, case `dev-0000` has answer `64285` and A0 generated
`' 64285\nAnswer: 64285\nAnswer: 64285'` — correct on the first token, then
looping to the 24-token cap. Full-string equality scores that 0.

The cause is the prompt contract, not the arms. This checkpoint is chat-shaped
(`<|im_end|>` = 32744 is its declared EOS), and the registered P3 protocol is
`"raw completion, exact token lengths, no truncation or changed parser"`. A raw
completion never reaches a turn boundary, so no stop token is ever the argmax.
Widening the stop set to the checkpoint's full declaration `[32742, 32744]`
(fixed earlier this session) did not help: **neither** token is ever emitted.

**The endpoint was not changed after seeing this.** Rescoring with a
prefix/containment rule now would be endpoint shopping. The fix belongs in a
preregistered amendment (below), applied to a rerun.

---

## 2. What DOES stand

### 2a. Identity gates hold token-for-token at real lengths (strong)

| pair | identical generations |
|---|---|
| A0_global == A4_global_gate | **72 / 72** |
| A2_local512 == A5_local512_gate | **72 / 72** |
| A3_sink512 == A6_sink512_gate | **72 / 72** |

Bit-exact agreement across 720 real generations at 512-2048 tokens, far
stronger than the 17-token mechanical check in P2. `2*sigmoid(0) = 1.0` is
exactly identity in float32, and the untrained gate provably changes nothing.
This is a precondition for the future trained-gate comparison and it is now met
on real data.

### 2b. Divergence tracks the predicted eviction map exactly

The window rule predicts, before the run, which cells lose the answer from the
local window. Comparing each arm's generations against A0 token-for-token
(**diagnostic only, not a registered endpoint**):

| arm | predicted-OUT cells that differ from A0 | in-window cells that differ |
|---|---|---|
| A3_sink512 (W512, S4) | **40 / 40 (100%)** | 24 / 32 (75%) |
| A2_local512 (W512, S0) | **40 / 40 (100%)** | 29 / 32 (91%) |
| A7_sink1024 (W1024, S4) | **20 / 20 (100%)** | 30 / 52 (58%) |

Every case where the topology says the evidence must leave the local window
produced a different generation, in all three windowed arms. The mechanism is
doing exactly what the registered topology says, on the real model, at real
lengths. Overall divergence from A0 orders sensibly with severity:
A7 (50/72 differ) < A1 (57) < A3 (64) < A2 (69) < A9 (71) = A8 (71).

### 2c. Repetition, a registered metric, moves in the predicted direction

`max_token_repeat` is in the registered metric list. Descriptive only, no
significance claim, no paired test:

| arm | mean max_token_repeat |
|---|---:|
| A0_global | 4.53 |
| **A2_local512** (pure W512, no sinks) | **6.31** |
| **A3_sink512** (W512 + 4 sinks) | **4.42** |
| A1_local1024 | 4.31 |
| A7_sink1024 | 5.26 |
| A9_frontloaded512 | 4.85 |
| A8_all_local512 | 4.62 |

Pure 512-wide windowing raises looping well above the all-global reference
(6.31 vs 4.53); adding four prefix sinks at the same budget brings it back to
reference level (4.42). That is the direction the sink hypothesis predicts, on
a registered metric. It is one metric on 72 correlated synthetic cases with no
multiplicity control, so it motivates a rerun, nothing more.

---

## 3. Anomaly investigated, not ignored

`A8_all_local512` reports `elapsed_seconds = 9298.9` against ~185-200 s for
every other arm. A8 has **zero** global layers, so it does strictly less
attention work than A7 (192 s) and A9 (182 s) which ran either side of it; the
stall is external, and it coincides with an interruption of the controlling
session. `timing_causal` is registered `false` and no speed claim is made.
Correctness is unaffected: `validate_execution` binds observed score-rectangle
elements, attention matmul work and every layer's resident KV to the planner
for each of the 720 generations, and raises on any mismatch — the run passed,
so A8's arithmetic reconciles exactly like the rest.

---

## 4. What this establishes, and what it does not

**Establishes:** P3 executed under a valid, receipt-gated, current-code
qualification; untrained gates are exactly identity on real generations
(72/72, three pairs); the local/global mechanism changes behaviour precisely
where the registered topology predicts it must; a registered repetition metric
degrades under pure windowing and is restored by prefix sinks.

**Does not establish:** anything about capability, preservation or damage. The
registered capability endpoint produced no usable signal in any arm. **The
primary question — does frozen R2 tolerate the A3_sink512 retrofit — remains
unanswered.** Nothing here licenses a quality, speed, memory or training claim,
and P4/P5 stay closed.

---

## 5. PROPOSED readout amendment — NOT AUTHORIZED, NOT EXECUTED

Requires sign-off before any rerun. Written from the reference arm's behaviour
only, before any per-arm rescoring, so it cannot be tuned to a contrast.

The registered endpoint measures "emitted exactly the answer and stopped". The
model reliably does the first half and never the second, so the endpoint has no
resolution. Two independent fixes, and they should be reviewed together:

1. **Make termination reachable.** Either present the task through the
   checkpoint's own chat template so `<|im_end|>` is a live continuation (this
   changes the registered `prompt_protocol` and the `diagnostic_id`), or keep
   raw completion and register a stop on the first newline after the answer
   span. The first is truer to how the checkpoint is used; the second
   preserves the raw-completion invariant.
2. **Register a resolution-bearing endpoint.** Add `first_content_exact` — does
   the generation, trimmed, begin with the answer — as a *pre*registered
   primary alongside `trimmed_exact`, with `trimmed_exact` retained so the two
   can be compared. On the reference arm this recovers 47/72 of signal that the
   current endpoint discards.

Do not adopt only (2): an endpoint that tolerates non-termination would score a
looping arm and a clean arm alike, and `max_token_repeat` already shows the arms
differ in exactly that behaviour. Adopt both, rerun all ten arms, and report the
old and new endpoints side by side.
