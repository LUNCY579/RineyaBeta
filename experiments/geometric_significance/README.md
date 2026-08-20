# Geometry-aware significance: does diversity-aware eviction beat top-k?

Branch `exp/geometric-significance`. Self-contained; `scripts/train/` untouched.

---

## The idea, and the precise reading of it

> "Try if a min-max pump-dump on an imaginary graph would improve the cache
> selection and storage even better. Imagine if the words and weights are
> distributed on a 3D coordinate system, each time the significance evictions
> change something and update a coordinate, find the local max and min, project
> it on the significance matrix, and update the significance accordingly."

The reading that makes this precise and promising: **plain top-k selection is
redundancy-blind.** If many cached keys are near-duplicates and all score highly,
top-k spends many slots on one piece of information while a distinct
lower-scoring key is evicted. The retained-mass bound cannot see this, because
duplicated attention mass counts fully toward `rho`. Selecting **local maxima of
the significance field over a neighbourhood graph in key space** would instead
pick representatives of distinct modes — coverage rather than magnitude.

This is a coherent generalisation of something that already worked: **SnapKV
pooling is exactly this operation in 1-D along the position axis**, and it
measured +0.018 on rag_dev at keep 0.50. The proposal extends it from the
position axis to the key-embedding geometry.

The premise is therefore measurable, and it should be measured before any policy
is designed (P7).

---

## Phase 0 kill test: is the selected set actually redundant?

`src/redundancy.py`, 8 rag_dev prompts, budget B=64, 28 layers × 8 KV heads.
Post-RoPE keys, which is what attention actually multiplies against.

The control is a **random** B-subset of the same cache. If top-k is no less
diverse than random selection, top-k is not creating redundancy and a diversity
term has nothing to recover.

| metric | top-k | random | ratio |
|---|---:|---:|---:|
| mean pairwise cosine (raw) | 0.689 | 0.661 | 1.04 |
| effective rank of 64 (raw) | 4.79 | 5.83 | 0.82 |
| distinct clusters of 64 (raw) | 34.6 | 40.1 | 0.86 |
| mean pairwise cosine (**centred**) | 0.090 | −0.003 | — |
| effective rank of 64 (**centred**) | 17.07 | 20.83 | 0.82 |
| distinct clusters of 64 (**centred**) | **60.44** | **61.95** | **0.98** |

### The raw numbers are an artefact, and the artefact is itself a finding

Mean pairwise cosine across **all** keys is 0.661, and a *random* 64-key subset
has effective rank 5.83 of 64. The key space is strongly **anisotropic** — a
large shared mean direction makes any two keys look similar. So the eye-catching
"effective rank 4.79 of 64" is a property of the model's key geometry, not of the
selection, and roughly 91% of it would be there no matter which keys were chosen.

**Design consequence, and it is not optional:** a similarity threshold in raw key
space is close to meaningless when every pair sits at 0.66. Any geometric policy
must centre (or whiten) the keys first. After centring, a random subset has mean
cosine −0.003 — near-orthogonal, as a usable geometry should be.

### Verdict at reachable lengths

In centred space, **top-k keeps 60.4 of 64 slots distinct against 61.9 for random
selection — 2% worse.** Top-k is already about as diverse as chance allows. The
duplicate-slot waste that the geometric proposal exists to reclaim is, on this
model at these lengths, roughly 2% of the budget.

The effective-rank gap (0.82, consistent in both raw and centred space) is real
and reproducible. But it is a *soft* concentration effect — the selected keys lean
toward a shared subspace — not the hard duplicate-slot waste a non-maximum
suppression or clustering policy would reclaim. A soft diversity *penalty* could
act on it; a hard threshold policy has almost nothing to grip.

**This does not say the idea is wrong.** It says the premise it depends on is not
present in a form worth exploiting *here*.

---

## What would change the answer

1. **Long contexts.** These prompts are 41–330 tokens of dense, non-repetitive
   evidence. Redundancy is what eviction exploits, and a 232-token RAG prompt has
   almost none. A 32K context with repeated boilerplate, tables, or restated
   instructions could be far more duplicated — and that is precisely the regime
   this repository cannot reach (nothing exceeds ~2.7K tokens, see
   `docs/EXPERIMENT_DESIGN_KV_LONGCTX.md` §2a). This is now the third line of work
   blocked on the same missing corpus.
2. **A soft diversity term rather than hard suppression.** The 0.82 effective-rank
   ratio is the part that is really there. A penalty proportional to similarity
   with already-selected keys (a submodular / DPP-style objective) targets that,
   where a cosine threshold does not.
3. **Larger budgets.** At B=64 of ~232 positions the selection is 28% of the
   cache. Redundancy pressure should rise as the budget tightens; B=16 or 32 is
   worth measuring before concluding.

---

## Cost reality check

Whatever the policy, it competes against a hard measured fact: **a 16× smaller
cache already buys only 1.15× decode**, because per-step top-k and gather across
28 layers eat the bandwidth saving. Any geometric policy adds pairwise-similarity
work on top of that — `O(B²d)` per layer per head at minimum. At B=64, d=128,
8 heads, 28 layers that is ~1.2 GFLOP per eviction, which is not free at a
per-step cadence.

So the bar is high: a geometric policy must recover meaningfully more than the
~2% diversity headroom to pay for itself, and it must be amortised (via `slack`)
rather than run every step.

---

## Status

- [x] Phase 0 kill test — premise measured, largely unsupported at reachable lengths
- [ ] Formalization review (4 independent readings + adversarial critique) — in flight
- [ ] Implementation, conditional on the above

Files:

```
src/redundancy.py       Phase 0: redundancy of the selected key set, with a
                        random-subset control and a centred-space view
results/redundancy.json measured, per layer
```
