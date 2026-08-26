# Quarantined 2026-08-26: measured with a flooded `grow_spans` gate

`grow_spans` computed its noise floor as the median of a vector with `0.0`
substituted for every masked slot. `select_keep` masks sinks and the recent
window to `-inf` before calling it, and on this arm's geometry (budget 58,
sinks 4, recent 31) that is **59% of the vector** — so the floor was exactly
`0.0`, and since scores are non-negative the gate `finite > 0 * mult` opened on
**every** eligible position: 24 of 24, where the true median admits 2.

Growth then floods the stream and the top-k after it is arbitrary. This arm
therefore measured a degenerate positional policy, not derived reach.

**Numbers that came from here and must not be cited:**
- `mean_quality_score = 0.33833`
- the paired contrast `keep0.25_causal_vs_derived = -0.07452` (p = 5.52e-12),
  and with it the claim that derived reach loses on language **at both keep
  fractions tested**. The keep-0.50 leg (-0.03513) was measured at 30% masked —
  degraded but not collapsed — and is a separate question.

Fixed in `scripts/kv/significance.py` (nanmedian over valid entries) and pinned
by four new checks in `scripts/kv/test_grow_spans.py`. Re-run before quoting.
