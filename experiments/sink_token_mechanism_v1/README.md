# STM-1: what do persistent sink tokens contribute?

This isolated follow-up starts from corrected head `efb05337d55491478ddd1aa92550f0db18ce5e4f`.
It does not repeat or reinterpret the zero-power global-layer comparisons.

The inherited effect is narrow but strong: keeping the first four ordinary
prompt-token KV rows in every W512 local layer improved A8 to A10 by 0.153 (11
wins, 0 losses, exact p=9.8e-04 after Holm), the third one-sided replication.
Those rows are not new learned register parameters. STM-1 asks whether their
benefit comes from stable position, key-based routing/softmax capacity, value
information, or dose.

Eleven frozen arms form four primary causal contrasts, a visibility control and a secondary count curve.
All arms use zero global layers and exactly the same W512 total key budget.
`zero_k` retains values while destroying learned key routing; `zero_v` retains
the attention denominator but removes sink information; `permute_v` breaks the
key/value identity pairing; `mean_v` preserves only an aggregate reservoir;
`interior4` moves persistent rows away from the prefix. Attention mass, entropy,
sink-value contribution, logit KL and output verdicts are recorded together.

Run CPU mechanics with:

```bash
python -m unittest sink_mechanism_v1.test_interventions -v
python -m sink_mechanism_v1.verify
```

Passing these checks opens only the right to perform inherited checkpoint/source
qualification. Frozen open DEV may run only when every provenance, native-oracle,
streaming-trace, exact-budget and diagnostic-finiteness gate passes at one code
identity. Training, sealed evaluation, architecture promotion, efficiency and
production claims remain closed.
