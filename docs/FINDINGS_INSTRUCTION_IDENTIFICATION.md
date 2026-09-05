# Selection is saturated, protection is separate, and pooling hid both

Preregistered in [`PREREG_INSTRUCTION_IDENTIFICATION.md`](PREREG_INSTRUCTION_IDENTIFICATION.md),
frozen before either new arm ran. Analyser:
`scripts/design/analyse_instruction_identification.py --check`. Data:
`experiments/kv_significance/results/instruction_identification.json`.

All four coverage gates passed for both new arms: 600/600 cases,
`heavy_zero_count == 0`, shared-manifest diff exactly `[elapsed_s, select, tag]`,
and instruction protection demonstrably fired (abstain 0.9524 and 0.9405 against
the comparator's 0.9226, where every unprotected arm scores 0.0000).

## 1. Why this cell was run

Three arms already on disk, differing in the `select` field alone, appeared to
decompose the programme's headline as follows:

| arm | selector | instructions kept | pooled delta vs uncompressed |
|---|---|---|---|
| `c7-oracle-k0.05-n600` | oracle: the gold record | no | −20.17 pp |
| `c7-docbm25-k0.05-n600` | doc-BM25 | no | −20.50 pp |
| `c8-docbm25sys-k0.05-n600` | doc-BM25 | yes | +5.17 pp |

Read pooled, that says replacing BM25 with a *perfect retrieval oracle* is worth
0.33 pp while instruction protection is worth 25.67 pp — i.e. the selector is
decoration. **That reading is wrong**, and it was recorded as the leading
hypothesis before this cell was run. It is wrong because both unprotected arms
score exactly 0.0000 on the abstain stratum, which floors them regardless of
selection quality and drowns the answer stratum in the pooled average.

The campaign could not settle it, because every `-sys` arm among the 49
`rag_long4k` arms was a BM25 variant: there was no `random-doc-sys` and no
`oracle-support-sys`. So "selection contributes nothing **given** protection" had
never been tested, only inferred from the row where nothing could show.

## 2. The answer-stratum ladder

The preregistered primary stratum, n = 432, endpoint
`contract_fulfillment_score >= 0.999`.

| arm | selector | protected | answer | abstain | pooled |
|---|---|---|---|---|---|
| `c19-randomdocsys` **NEW** | random record draw | yes | **0.0486** | 0.9524 | 0.3017 |
| `c7-off-n600` | — (uncompressed) | — | 0.7708 | 0.8214 | 0.7850 |
| `c7-docbm25-k0.05` | doc-BM25 | no | 0.8056 | 0.0000 | 0.5800 |
| `c8-docbm25sys-k0.05` | doc-BM25 | yes | 0.8032 | 0.9226 | 0.8367 |
| `c7-oracle-k0.05` | oracle | no | 0.8102 | 0.0000 | 0.5833 |
| `c19-oraclesys` **NEW** | oracle | yes | 0.8009 | 0.9405 | 0.8400 |

A sanity check that the floor arm behaves: `random-doc-sys` scores **0.0486** on
the answer stratum at `keep_frac` **0.05**. A uniform record draw recovers the gold
record at chance, and the endpoint tracks it to three decimal places.

## 3. Both preregistered contrasts

**A vs C — does selection matter given protection?**
Answer stratum: **−75.46 pp, 0W/326L, p = 1.5e−98.**
Selection matters enormously. The verdict is **SELECTION CONTRIBUTES**.

**B vs C — how much room is left above BM25?**
Answer stratum: **−0.23 pp, 1W/2L, p = 1.00.**
None. A perfect retrieval oracle with the same protection is indistinguishable
from a blind BM25 index. The verdict is **SATURATED**.

These two together are the finding. Selection spans 76 pp between a random draw
and an oracle; BM25 already sits at the top of that span. The earlier claim in
`FINDINGS_COMPRESSION_IMPROVES_QUALITY.md` that "the blind selector is at 99% of
the oracle's ceiling" is correct on the answer stratum and is now measured against
a real floor rather than asserted against none.

## 4. What this does and does not license

**Licensed.** On this corpus, at this budget: record selection is necessary and
BM25 exhausts it; instruction protection is separately necessary and is what makes
the abstain stratum work at all; the two are independent and act on different
strata. Compression at 16.5x with either selector plus protection beats the
uncompressed model on the answer stratum — C +3.24 pp (15W/1L, p = 5.2e−04),
B +3.01 pp (14W/1L, p = 9.8e−04).

**Not licensed, and the pooled numbers must stop being quoted alone.** The pooled
+5.17 pp is roughly half an artifact of the abstain endpoint: the whole abstain
gain is one 38-case category whose contract is a substring test for a canned
sentence (see [`audit_abstain_endpoint.py`](../scripts/design/audit_abstain_endpoint.py)
and the correction in `FINDINGS_COMPRESSION_IMPROVES_QUALITY.md`). Any pooled
figure in this programme sums a real answer-stratum effect with an artifactual
abstain-stratum one, and the two have opposite relationships to the truth.

**Bounded by construction.** `make_long_rag.py` excludes same-topic distractors,
so BM25 ranks the gold record first in 530/530 answerable cases. "Selection is
saturated" is therefore a statement about **this corpus**, where retrieval is easy
by design. It does not follow that BM25 saturates selection on a corpus whose
distractors share the gold record's topic, and the cheapest next experiment is a
corpus that does not exclude them.

## 5. Cost

Two arms, 1100.7 s and 1002.4 s on one RTX 4070 8 GB, run one at a time. No arm
was re-run: C and the reference were reused from the committed campaign.
