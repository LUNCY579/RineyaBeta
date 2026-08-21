# Superseded 128K ranking runs — kept, not deleted

These three files were the first attempt at comparing sum-ranking against
mean-ranking at 128K. They are superseded by the five matched arms
`../pk_128k_e{0.0,0.25,0.5,0.75,1.0}.json`, and they are kept because a claim was
published from them and the retraction should be checkable.

| file | what it actually is | why it was retired |
|---|---|---|
| `pk_128k_sum.json` | 3 trials, depths 0.5 and 0.9 only | too few trials, and missing depth 0.1 |
| `pk_128k_mean.json` | 3 trials, depths 0.5 and 0.9 only | same |
| `pk_128k_final.json` | 6 trials, all three depths, mean ranking | superseded by `pk_128k_e1.0.json` |

**The claim that came out of these and is now withdrawn.** A table circulated
reading `sum 1.00 / 0.33 / 0.00` against `mean 0.17 / 0.17 / 0.83`, described as
the two endpoints failing in opposite directions. The sum column came from a
3-trial run (`../pk_long_aware.json`) and the mean column from a 6-trial one — a
comparison between different trial counts and different depth sets. Re-run
matched at 6 trials × 3 depths, the sum arm is **0.33 / 0.17 / 0.17**, not
1.00 / 0.33 / 0.00.

None of these files records which arm produced it: they predate the manifest
fields `count_exponent`, `normalize`, `query_aware` and the per-row
`rank_exponent`, which exist because of this. `pk_128k_final.json` could only be
identified as the mean arm after the fact, by `pk_128k_e1.0.json` reproducing it
exactly (0.17 / 0.17 / 0.83) — which is also the cleanest evidence available that
the harness is deterministic given the same flags.
