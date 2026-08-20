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

## Phase 1: the surviving formalization, and its death

Design review (four independent formalizations, each adversarially reviewed for
novelty and implementability) falsified three of the four on measurements of the
real model:

| formalization | embedding | premise measured | verdict |
|---|---|---|---|
| BT-KV | attention logits | redundant fraction median **0.0000**, top-k overlap 0.9994 | dead |
| MoSS | value-vector cosine | 0.089 mean partners; 95.5% of candidates have none | dead |
| CoverKV | facility location on keys | dominance 0.576 vs random 0.521 | subsumed by SnapKV pooling |
| **RSS-P** | **attention profile** | cosine field median 0.383, P99 0.933 — heavy-tailed | survived |

RSS-P is the interesting one, and it directly answers the objection raised by the
Phase-0 measurement above. Embedding positions by their *key vector* is hopeless
because the key space is anisotropic. Embedding them by their **attention
profile** — the softmax row a position receives from the observation window — is
not: that field is genuinely heavy-tailed, and the embedding is free, because the
attention weights are already materialised while computing the scores.

### Gate 0: does suppression improve the attention output?

`src/rss_probe.py`. Held-out decode queries (scores built from the prefill
window, error measured on the queries of subsequently generated tokens, because
importance persistence is only 0.584). Exact attention error
`||o_full - o_hat|| / ||o_full||` over 23,296 (layer, head, step) samples.

The control is **rss-shuffle**: permute the candidate→profile-row assignment
before the Gram. That preserves the edge count, the edge-weight multiset and the
number of slots moved exactly, and destroys only the pairing between a key and
its claimed duplicates. Without it, neither sign is attributable.

| τ | slots moved | topk | rss | rss-shuffle | rss-mass |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 29.4 | 0.1424 | 0.1903 (**+33.7%**) | 0.1668 (+17.2%) | 0.1832 |
| 0.93 | 27.0 | 0.1424 | 0.1781 (+25.1%) | 0.1594 (+11.9%) | 0.1666 |
| 0.96 | 16.7 | 0.1424 | 0.1605 (+12.7%) | 0.1495 (+5.0%) | 0.1515 |
| 0.99 | 3.8 | 0.1424 | 0.1469 (+3.2%) | 0.1426 (+0.2%) | 0.1423 |

(lower error is better; percentages are relative to top-k)

**KILL-1 at every τ.** Three things make this a real result rather than one bad
hyperparameter:

1. **Monotone.** Damage scales with the number of slots the policy moves, and
   converges to zero only as τ→1 — that is, as the policy converges to plain
   top-k. The best available outcome is "harmless because it does nothing".
2. **Worse than its own matched-random control at every τ.** Suppression is not a
   neutral perturbation of the cutoff; it is *worse than randomly perturbing the
   same number of slots*. The profile-similarity signal is anti-correlated with
   what should be kept.
3. **Retained mass falls with it** (rho 1.861 → 1.791 as suppression rises),
   which is the mechanism showing up directly in the quantity the bound predicts.

### Why the sign is negative — the useful part

Positions with similar attention profiles are **not redundant, they are
corroborating**. If two positions are attended in the same pattern by the same
query heads, they are jointly participating in one computation, and deleting
either one damages it. Attention-profile similarity measures *functional
co-participation*, not *informational duplication* — so a policy that suppresses
near-duplicates in that space is systematically deleting the wrong things, which
is exactly why it loses to a random reshuffle of equal magnitude.

That also explains the Phase-0 result. There was no exploitable duplication in
key space, and the one space where the similarity field looked promising turns
out to encode the opposite of what the idea assumed.

**Mass compensation** — a surviving leader inheriting its displaced followers'
weight via an additive logit bias — was the one variant an independent reviewer
measured to beat top-k elsewhere (−2.1%). It does not reproduce here: it tracks
rss at every τ and reaches parity only at τ=0.99, where it moves 3.8 slots and
does essentially nothing. Recorded as a discrepancy rather than resolved.

---

## Verdict

The geometric idea is **dead on this model at these lengths**, for a
comprehensible reason rather than a measurement failure. Two independent kill
tests, each with the correct control:

- **Phase 0** — in centred key space, top-k already keeps 60.4 of 64 slots
  distinct against 61.9 for random selection. No duplication to reclaim.
- **Gate 0** — in attention-profile space, where duplication *does* appear to
  exist, suppressing it makes the attention output monotonically worse, and worse
  than a matched-random control.

Total cost: two probes, about four minutes of GPU time, no quality sweep.

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

- [x] **Phase 0** — key-space redundancy measured; premise unsupported (2% headroom)
- [x] **Phase 1 design review** — 4 formalizations, 8 adversarial reviews; 3 falsified
- [x] **Gate 0** — profile-space suppression measured; KILL-1 at every τ
- [ ] ~~Implementation~~ — not reached, and correctly so

```
src/redundancy.py         Phase 0: redundancy of the selected key set, with a
                          random-subset control and a centred-space view
src/rss_probe.py          Gate 0: exact attention error under four policies,
                          on held-out decode queries, with a matched-random control
results/redundancy.json   measured, per layer
results/rss_probe*.json   measured, per tau
```

Reproduce:

```bash
python src/redundancy.py --cases 8 --budget 64
python src/rss_probe.py --cases 64 --keep 0.5 --tau 0.93
```

## What this branch is for

Keeping a negative result that cost four minutes instead of the ~40 minutes a
quality sweep would have taken, and the design review that made it possible to
know which four minutes to spend. The branch is not merged; it is the record of
an idea that was worth testing and did not survive contact with the model.
