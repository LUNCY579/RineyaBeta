# Preregistration: can the endpoint tell a correct answer from a correct substring?

Frozen 2026-09-04, before any reply is adjudicated. Branch `research/cornerstone2-v1`.

## 1. The threat this settles

One claim in this programme still stands: on the answer stratum, compression at
16.5x beats the uncompressed model by **+3.24 pp** (15W/1L, p = 5.2e-04, cluster
CI [+1.38, +5.28]). Everything else has been withdrawn, restated, or bounded.

That claim rests on `contract_fulfillment_score >= 0.999`, which is a **substring
test**: the reply scores 1 if it contains the gold phrase. It cannot distinguish

    "The launch date for PineRelay 1225 is June 9, 2032."          (correct)
    "Zhe protocol for CobaltHub 1238 is BLE 5.4"                   (correct token, corrupted)
    "Thread: none Protocol: BLE 5.4"                               (correct token, not an answer)

and the compressed arm's wins look systematically like the second and third kinds.
On the 17 answer-stratum cases where `c9-sys-B256` beats uncompressed, mean
`intelligibility_score` is **0.3324 against 0.6265**, and **11 of 17** winning
replies carry `quality_hard_fail` against **6 of 17** for the uncompressed reply.

Gating on `quality_hard_fail` collapses the claim to +0.69 pp (6W/3L, p = 0.51),
but that gate is not admissible as the decisive test: it fires on a third of
uncompressed replies and its commonest trigger is digit-ratio noise, so it partly
just means "the answer contains a date".

A regex detector written by whoever knows which arm they want to win is not
admissible either. So the instrument is **blind adjudication**.

## 2. Sample

Fixed before adjudication, built by
`scripts/design/build_adjudication_sample.py` (deterministic, seed 20260826).

Arms: `c7-off-n600` (R, uncompressed), `c8-docbm25sys-k0.05-n600` (C, headline),
`c9-sys-B256` (N, the B=256 variant). Answer stratum only, n = 432.

* **Disputed set: 21 cases.** The union of cases where C differs from R, or N
  differs from R, under the committed endpoint. 16 and 19 respectively.
* **Concordant set: 40 cases**, drawn by seeded permutation from the 330 cases
  where all three arms score 1 under the committed endpoint.

Each case contributes all three arms' replies: **183 replies total.**

Each reply is emitted with its question and its gold answer, with **arm identity
stripped**, under an opaque `item_id`, in an order fixed by a seeded permutation
so that neighbouring items are not the same case. The mapping from `item_id` to
(case, arm) is written to a separate file that is not shown to any adjudicator
and is not read until every label is in.

## 3. Rubric

Each reply gets exactly one label, judged against the question and the gold
answer alone:

| label | meaning |
|---|---|
| `CORRECT_CLEAN` | states the gold answer, as a well-formed reply a user could act on |
| `CORRECT_DEGRADED` | contains the gold answer but the reply is garbled, truncated mid-thought, mislabels the entity, or is a fragment rather than an answer |
| `WRONG` | asserts something other than the gold answer |
| `ABSTAIN` | declines, or says the information is unavailable |
| `UNINTELLIGIBLE` | cannot be read well enough to classify |

Three independent adjudicators per reply. The label is the **majority of three**;
where all three differ, a fourth adjudicator chooses among the three labels
already given.

## 4. Endpoints

* **Primary (strict): `CORRECT_CLEAN` = 1, everything else = 0.** This is the
  quantity the programme has been claiming and has never measured.
* **Secondary (permissive): `CORRECT_CLEAN` or `CORRECT_DEGRADED` = 1.** This is
  what the substring endpoint is *trying* to compute.

## 5. Validity gates

No verdict is emitted unless all hold:

1. **Adjudication is not noise.** Unanimity or 2-of-3 majority on >= 90% of the
   183 replies; the count of 3-way splits is reported.
2. **Adjudicators track the contract.** On the 120 concordant-set replies, the
   *permissive* endpoint agrees with the committed endpoint on >= 90%. If
   adjudicators disagree with the substring test on cases nobody disputes, they
   are measuring something else and the exercise is void.
3. **The paired test can conclude.** >= 6 discordant pairs under the primary
   endpoint; otherwise the exact McNemar power ceiling exceeds 0.05 and the cell
   cannot return a result whatever the labels say.

## 6. Decision rule

Applied to C vs R on the primary endpoint, over the disputed set.

* **SURVIVES.** Delta > 0 and p <= 0.05. The answer-stratum claim is real: the
  gain is correct, legible answers, not substrings. It becomes the programme's
  headline, stated on the answer stratum only.
* **ENDPOINT-DEPENDENT.** p > 0.05 and |delta| < 2.0 pp. The claim is withdrawn
  as an accuracy claim and restated: compression at 16.5x does not degrade
  answer accuracy, and the apparent gain was the endpoint crediting degraded text
  that contains the gold token. This does **not** close the programme; the memory
  and floor results are untouched.
* **INDETERMINATE.** Anything else, including failing gate 3. No verdict. Report
  the achieved MDE.

## 7. The conditioning problem, and how it is bounded

The disputed set is defined **by the committed endpoint**, which is the instrument
under suspicion. Restricting to it conditions on that instrument, so the primary
analysis is biased by construction if adjudication also flips cases the committed
endpoint calls concordant.

The 40-case concordant sample exists to bound that. From it, the rate at which
adjudication creates new discordance among all-correct cases is estimated with a
Clopper-Pearson interval and projected onto the remaining 290 unadjudicated
concordant cases. If the projected number of induced discordant pairs could
change the sign of the primary contrast at the upper bound of that interval, the
verdict is **INDETERMINATE regardless of the primary p-value**, and the honest
statement is that the full stratum must be adjudicated.

This is stated now because it is the most likely way this design fails, and
discovering it afterwards would look like an excuse.

## 8. What this does not test

Generality. One model, one corpus, one prompt template. `make_long_rag.py`
excludes same-topic distractors, so BM25 ranks the gold record first in 530/530
answerable cases; nothing here speaks to a corpus where retrieval is hard.
