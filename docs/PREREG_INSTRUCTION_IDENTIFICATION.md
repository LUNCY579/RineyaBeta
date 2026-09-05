# Preregistration: does record selection do anything once the instructions are kept?

Frozen 2026-09-04, before either new arm is run. Branch `research/cornerstone2-v1`.

## 1. Why this cell exists

The programme's headline is that BM25 record selection plus instruction protection
beats the uncompressed model by +5.33 pp at 16.5x cache reduction. Three arms
already on disk, differing in the `select` field alone and identical on model,
budget, `keep_frac`, sinks, recent, pool, seed and protocol (verified by manifest
diff), decompose that number:

| arm | selector | instructions kept | delta vs `c7-off-n600` |
|---|---|---|---|
| `c7-oracle-k0.05-n600` | oracle: the gold record | no | **-20.17 pp** (18W/139L) |
| `c7-docbm25-k0.05-n600` | doc-BM25 | no | **-20.50 pp** (15W/138L) |
| `c8-docbm25sys-k0.05-n600` | doc-BM25 | yes | **+5.17 pp** (34W/3L) |

Replacing BM25 with a **perfect retrieval oracle** moves the endpoint by 0.33 pp.
Adding instruction protection moves it by 25.67 pp. On the evidence committed
today, the headline is a property of not evicting the system prompt, and the
selector is decoration.

But that is not identified. Every `-sys` arm in the campaign is a BM25 variant --
of 49 `rag_long4k` arms there is no `random-doc-sys`, no `oracle-support-sys`. So
"BM25 selection contributes nothing **given** protection" has never been tested;
it is inferred from the no-protection row, where both arms are floored at -20 pp
and could not show a difference even if one existed.

This cell tests it directly, and it is the cheapest decisive experiment available
to the programme: two arms, about 20 minutes of GPU each.

## 2. Arms

Two new, two reused. Nothing is re-run that already exists.

| id | select | status |
|---|---|---|
| **A** | `random-doc-sys` | NEW -- granularity-matched random record draw, plus protection |
| **B** | `oracle-support-sys` | NEW -- gold record only, plus protection. A CEILING, not a control |
| **C** | `lexical-doc-bm25-sys` | reused: `c8-docbm25sys-k0.05-n600` |
| **R** | eviction off | reused: `c7-off-n600` |

`oracle_support_weights` documents itself as a mechanism probe that "may never be
reported as a control or compared to the mechanism as though it were one". That
restriction is honoured here: B is used only to bound how much selection could
possibly be worth, never as a baseline anything is claimed to beat.

**Matched factors**, to be verified by manifest diff before any analysis: `model`,
`adapter`, `dataset`, `budget`, `keep_frac` (0.05), `sinks` (4), `recent_frac`
(0.15), `pool` (7), `pool_back` (6), `pool_fwd` (0), `window` (16), `seed`
(20260826), `cap_mib` (6500), `protocol`, `case_count` (600), `eviction` (true).
The diff against `c8-docbm25sys-k0.05-n600` must show differences in **`tag`,
`select` and `elapsed_s` only**. Any other difference voids the cell.

## 3. Endpoint

`contract_fulfillment_score >= 0.999`, binarised, matching
`analyse_capacity_floor.py:33`.

**Primary stratum: the answer stratum, n = 432.** This is a change from the
programme's pooled endpoint and the reason is recorded here rather than discovered
later: on the abstain stratum the contract is two `any_phrases` rules over one
literal sentence, the entire committed abstain gain is the single 38-case
`partial_support` category at +44.74 pp, and 12 of its 17 wins are cases where the
uncompressed model had already abstained in different words
(`scripts/design/audit_abstain_endpoint.py`, committed 2026-09-04). A pooled
endpoint here would measure canned-string emission, not accuracy.

Pooled and abstain are reported as **secondary**, always beside the primary, never
alone.

## 4. Test

Paired exact two-sided McNemar on discordant pairs, matched by case id.
Reported with each contrast: n paired, W/L, exact p, the conditional 95% CI on the
paired difference, and the **power ceiling** `2 * 0.5^n_disc` from
`scripts/design/audit_power_ceiling.py`.

Because the 600 cases come from 63 distinct `topic_id`s with evidence documents
reused roughly 65 times each, a **topic-level cluster bootstrap** (10,000 resamples
over topics, not cases) is reported beside every p-value. Where the two disagree,
the cluster interval governs.

Contrasts, in this order:

1. **A vs C** -- the identification contrast. Does BM25 beat a random record draw
   when both keep the instructions?
2. **B vs C** -- the ceiling. Does a perfect oracle beat BM25 when both keep the
   instructions?
3. A vs R, B vs R, C vs R -- each against uncompressed, for the record.

## 5. Decision rule

Fixed now, before any data exists.

- **SELECTION IS INERT** if, on the answer stratum, A vs C has p > 0.05 **and**
  |delta| < 2.0 pp **and** the 95% CI excludes a 5 pp advantage for C.
  Consequence: the claim "BM25 record selection improves accuracy" is **withdrawn**
  from `FINDINGS_COMPRESSION_IMPROVES_QUALITY.md`, the result is restated as
  instruction retention under compression, and the selector is described as
  interchangeable at this budget.
- **SELECTION CONTRIBUTES** if C beats A on the answer stratum at p <= 0.05. The
  contribution size becomes the honest magnitude of the retrieval claim, replacing
  +5.33 pp, and B vs C bounds how much more a better selector could buy.
- **INDETERMINATE** if p > 0.05 but the CI does not exclude a 5 pp advantage.
  **No verdict is emitted.** The achieved MDE is reported and the cell is either
  re-run at larger n or left open. A null from an underpowered cell closes nothing.

An inert result does **not** close the programme. It relocates the finding: keeping
a small, lexically-unrelated span that the query cannot retrieve is exactly what a
lexical index cannot do, which is the open question O14 already names. It would
make the mechanism claim smaller, more specific, and testable on any model.

## 6. Minimum coverage

No verdict of any kind is emitted unless **all** hold:

1. Both new arms complete **>= 594 / 600** cases (99%).
2. `heavy_zero_count == 0` in both new manifests.
3. The manifest diff against C differs in `tag`, `select`, `elapsed_s` only.
4. Instruction survival for A and B, measured by
   `scripts/design/instruction_survival.py`, is **>= the rate C achieves**. If
   protection did not actually fire, the cell measures nothing.

If a run aborts, its partial artifact is **deleted, not analysed**.

## 6a. Amendment, 2026-09-04, after arm A ran and before its result was read

Section 2 says the manifest diff against C "must show differences in `tag`,
`select` and `elapsed_s` only. Any other difference voids the cell." Arm A's diff
also shows `sinks_flag`, `protect_instructions`, `chunk_tokens` and `sinks`.

That wording was stricter than its intent and would void a valid cell. The three
extra fields **do not exist in C's manifest at all**: they were added to the
manifest schema after `c8-docbm25sys-k0.05-n600` ran, so `c8.get(k)` returns
`None` for each. Arm A's manifest is a strict superset of C's. `sinks` differs as
`4.0` against `4`, the same value written by a later code path.

Restated, and this is the rule the analyser enforces: **among keys present in both
manifests, the only permitted differences are `tag`, `select` and `elapsed_s`, and
numeric fields compare by value rather than by repr.** Verified for arm A: every
shared field is equal, including `budget`, `keep_frac`, `sinks`, `recent` (31.86),
`recent_frac`, `pool`, `pool_back`, `pool_fwd`, `window`, `seed`, `cap_mib`,
`model`, `adapter`, `protocol`, `case_count` and `mean_kept_ratio` (0.05).

Disclosed rather than silently applied, because the amendment was written after
the run existed. It was written before any outcome was computed.

## 7. Hardware discipline

One RTX 4070 8 GB shared with the desktop. `nvidia-smi
--query-compute-apps=pid,process_name --format=csv` is checked immediately before
each launch and no second job is started while one is live. Arms run **one at a
time**, A before B. Results stream per case, so an interrupted run leaves valid
partial data that is nonetheless deleted rather than analysed, per section 6.

## 8. What this does not test

- Whether the effect holds on any other model. One checkpoint,
  `artifacts/rineya-core-en32k-28L`, an in-house ~0.3B model fine-tuned on this
  corpus family.
- Whether it holds on a corpus whose distractors share the gold record's topic.
  `make_long_rag.py` excludes same-topic distractors by construction, so BM25
  ranks the gold record first in 496 of 530 answerable cases and this cell cannot
  distinguish "selection is inert" from "selection is saturated".
  That confound is real and is **not** repaired here; it is why an inert result
  must be written as inert *on this corpus*.
- Whether the answer-stratum endpoint can tell a correct answer from a correct
  substring inside degraded text. It cannot, and that remains open.
