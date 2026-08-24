# SGQR Negative Results

No empirical negative result is recorded at branch initialization.

Preserve future negative findings, including:

- a candidate layer whose INT4 remnant cannot recover deletion damage;
- significance equal to or worse than random;
- magnitude ranking matching significance;
- large group interactions that defeat independent top-k selection;
- state-conditioned routing worse than a static route;
- useful quality only at full activation;
- reference partial execution slower than dense execution;
- packed storage savings without VRAM savings;
- or model/domain-specific calibration failure.

Do not delete a negative result after retuning. Record the original configuration,
root cause where known, and whether a corrected follow-up changes the conclusion.
