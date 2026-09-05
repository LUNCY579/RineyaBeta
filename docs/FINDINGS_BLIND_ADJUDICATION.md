# The gain is real answers, not real substrings: +2.55 pp under blind adjudication

Preregistered in [`PREREG_BLIND_ADJUDICATION.md`](PREREG_BLIND_ADJUDICATION.md),
frozen before any reply was labelled. Analyser:
`scripts/design/analyse_blind_adjudication.py --check`. Data:
`experiments/kv_significance/results/blind_adjudication.json`.

## 1. What was at stake

After the adversarial review, exactly one claim was still standing: on the answer
stratum, BM25 + instruction protection at a 256-token budget beats the
uncompressed model by **+3.24 pp** (15W/1L, p = 5.2e-04). Everything else had been
withdrawn, restated or bounded.

That claim was denominated in `contract_fulfillment_score >= 0.999`, a **substring
test**. It cannot separate a correct answer from a correct token sitting inside
garbled text, and the compressed arm's wins looked systematically like the latter:
on the 17 winning cases the winning reply's mean `intelligibility_score` was
**0.3324 against 0.6265**, and **11 of 17** carried `quality_hard_fail` against 6.

So the last surviving claim rested on an instrument that had never been shown to
measure the thing it was standing for.

## 2. Instrument

**925 replies, arm identity stripped, three independent raters each, 2,775
labels.** Each reply was shown with its question and its gold answer and nothing
else, under an opaque `item_id`, in an order fixed by a seeded permutation. The
item-to-arm key was written to a separate file and not read until every label was
in. The sample builder refuses to emit if any item leaks an arm name; the analyser
refuses to run if two blocks share an item id or a case.

Five labels, separating `CORRECT_CLEAN` from `CORRECT_DEGRADED`. Majority of
three, with a fourth adjudicator reserved for three-way splits.

**Both instrument gates passed, and passed hard.**

* **923 of 925 items unanimous** across three raters. Zero three-way splits, so
  the tie-breaker never fired.
* On the 862 endpoint-concordant replies the permissive adjudicated endpoint
  agrees with the committed substring endpoint **860/862 = 99.8%**. The raters
  are tracking the response contract, not some private standard of their own.

## 3. Result

Every case in the answer stratum is adjudicated — 21 disputed and all 411
endpoint-concordant — so this is unconditional. Nothing is extrapolated: the
preregistered conditioning bound collapses to the **observed 6** induced
discordant pairs against a primary net of 7, with no projection at all.

| endpoint | delta | W/L | p |
|---|---:|---:|---:|
| **primary — correct AND legible** | **+2.55 pp** | 12/1 | **0.0034** |
| secondary — correct, legibility ignored | +3.47 pp | 16/1 | 0.00027 |
| *committed substring endpoint, for comparison* | *+3.24 pp* | *15/1* | *5.2e-04* |

**The claim survives.** The threat that motivated the experiment is not what was
happening: the compressed arm's wins are replies that three blind raters
independently called correct and well-formed.

The magnitude is smaller than the substring endpoint reported. +3.24 pp was
measuring correctness-or-something-like-it; **+2.55 pp is correctness that a
reader would accept**, and that is the number this programme should quote.

The gap between the two adjudicated endpoints (+3.47 permissive against +2.55
strict) is the honest size of the degradation effect: compression does cost some
legibility, and roughly a quarter of the apparent gain is text that contains the
right answer badly.

## 4. What the substring endpoint got wrong, measured

The interesting part is *how* it is wrong, because it is not the way one would
guess.

**At detecting whether the answer is present at all, it is nearly perfect.**
Against the permissive adjudicated endpoint, over all 925 replies:

* **0 false positives.** There is no reply where the gold string appears and the
  raters judged the answer absent.
* **5 false negatives.** The gold answer conveyed in a different surface form and
  scored zero. Two of them sit in the 81 cases where *every* arm scored 0 — a
  class the first three adjudication blocks did not cover, added to the design
  precisely because a substring test can fail this way.

**What it actually gets wrong is answerhood, not presence.** 25 replies were
labelled `CORRECT_DEGRADED`: the gold string is genuinely there, and the reply is
still not something a reader would accept. **23 of those the committed endpoint
scored 1.** That is the whole of the gap between the permissive +3.47 pp and the
strict +2.55 pp, and it is the precise sense in which the original concern was
right — not that the endpoint hallucinates correctness, but that it cannot tell
an answer from a fragment containing one.

## 5. Cost, and what it did not settle

2,775 labels, no GPU, about eight minutes of wall-clock across four blocks.

It does not touch generality. One model, one corpus, one prompt template.
`make_long_rag.py` excludes same-topic distractors, so BM25 ranks the gold record
first in 530/530 answerable cases; nothing here speaks to a corpus where retrieval
is hard. It also says nothing about the abstain stratum, which remains a
canned-string indicator and whose gain remains withdrawn — see
[`FINDINGS_COMPRESSION_IMPROVES_QUALITY.md`](FINDINGS_COMPRESSION_IMPROVES_QUALITY.md).
