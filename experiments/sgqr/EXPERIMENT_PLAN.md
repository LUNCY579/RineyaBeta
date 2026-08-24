# SGQR Experiment Plan

## Objective

Evaluate the quantized-remnant foundation before introducing confidence.

The primary causal chain is:

\[
\text{full-precision MLP deletion}
\rightarrow
\text{frozen low-bit donor}
\rightarrow
\text{fixed-budget group selection by significance}.
\]

## Stage I — Correctness

1. Verify packed INT4 and INT8.
2. Verify Qwen MLP group additivity.
3. Verify all-off equals deletion.
4. Verify all-on equals full quantized remnant.
5. Verify inactive groups are not dequantized or multiplied.
6. Verify donor buffers are frozen.
7. Verify state-dict reload.
8. Reproduce dense model behavior on real prompts.

## Stage II — Donor viability

Scan candidate layers.

For low, moderate, and high deletion sensitivity, measure:

- deleted teacher KL;
- INT8-all teacher KL;
- INT4-all teacher KL;
- full-remnant utility;
- packed storage.

Advance only if at least one nontrivial layer has meaningful low-bit recovery.

## Stage III — Significance

At fixed group fractions, compare:

- random;
- magnitude;
- static significance;
- dynamic significance;
- shuffled significance;
- reversed significance.

Fractions:

\[
0,\;0.0625,\;0.125,\;0.25,\;0.5,\;0.75,\;1.0.
\]

## Stage IV — Quality and systems frontier

Use the unchanged Rineya generation and quality protocol on development sets.

Report:

- absolute quality;
- Recovery Ratio;
- category effects;
- active group fraction;
- packed bytes;
- peak allocated/reserved VRAM;
- synchronized latency;
- tokens per second.

## Stage V — Strong controls

If the core mechanism is promising, compare:

- wrong-layer donor;
- random frozen donor;
- same-byte LoRA;
- deletion plus ordinary healing;
- static frozen-weight fusion;
- multiple group sizes;
- multiple layers;
- local survivor adaptation.

## Advancement gates

- **A:** implementation invariants pass;
- **B:** INT4 or INT8 donor is viable;
- **C:** significance beats random;
- **D:** a useful partial-budget point exists;
- **E:** measured systems behavior is characterized.

## Sealed test policy

Do not use the sealed test until layer, bitwidth, group size, utility definition,
router, budget, and evaluation code are frozen in a committed manifest.
