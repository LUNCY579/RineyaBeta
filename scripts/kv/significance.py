#!/usr/bin/env python3
"""Significance-gated KV eviction for Qwen3.

Implements the design in THEORY.md:

    keep = sinks(first S)  union  recent(last R)  union  top-B by accumulated score

Three pieces, kept separate so each can be tested on its own:

  1. `ScoreRecorder` -- wraps the SDPA attention interface to compute a
     significance score alongside the fast kernel, never by asking the kernel for
     its internals. SDPA and flash do not materialise the n x n score matrix, so
     scores are computed with an explicit small matmul over an observation window
     of the last `w` query rows: [heads, w, n] instead of [heads, n, n]. At 16K
     with w=16 that is a 1024x reduction, and the main attention path keeps its
     `is_causal` fast path (which we measured is worth ~1.9x prefill).

  2. `SignificanceCache` -- a DynamicCache that carries a per-position score and
     reports *true* sequence length even after compression, so RoPE positions for
     new tokens stay correct rather than silently re-indexing.

  3. `compress()` -- the eviction itself. Each KV head keeps its own top-B set,
     so the cache stays rectangular [b, n_kv, B, d] and no ragged bookkeeping is
     needed. Cached keys already carry their RoPE rotation, so dropping and
     reordering positions is safe.

Scores are summed over the query heads sharing each KV head, because eviction
removes a position for the whole group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer

# layer_idx -> [batch, n_kv_heads, n_positions] accumulated significance
_SCORES: dict[int, torch.Tensor] = {}
# Matching observation counts. Scores are a running SUM, so a position that
# survived T compression rounds has been scored T times and a newcomer once.
# Ranking on the sum freezes the heavy set: whatever got in early stays in,
# regardless of merit. Measured with query-aware scoring at 32K, budget 2048: a
# needle in chunk 0 retrieved at 1.00 while one in chunk 4 retrieved at 0.33 --
# backwards, since the later needle is no less relevant, only less observed.
_COUNTS: dict[int, torch.Tensor] = {}
# layer_idx -> [batch, n_kv_heads, n_positions] TRIVIAL lexical weight, carried
# through eviction exactly like _SCORES and _COUNTS. This is not part of the
# mechanism: it is the control arm that asks whether the accumulated-attention
# score does anything a token-id overlap with the question could not. It must be
# gathered by the same `idx` on every eviction, because after the first one each
# KV head holds a DIFFERENT subset of positions and a mask indexed by original
# position would silently address the wrong tokens.
_LEXICAL: dict[int, torch.Tensor] = {}
_STATE = {"enabled": False, "window": 16, "patched": False, "lexical_seed": None}

LEXICAL_SELECTS = ("lexical-idf", "lexical-naive", "lexical-doc-bm25",
                   "lexical-doc-bm25-sys", "lexical-chunk-bm25",
                   "lexical-chunk-bm25-sys")
# Arms that need a per-case weight vector installed by the caller. `random-doc` is
# the GRANULARITY-MATCHED floor: the same record segmentation as lexical-doc-bm25
# with the BM25 signal replaced by one uniform draw per record. Without it,
# lexical-doc-bm25's advantage is measured against a floor matched to the 7-token
# contiguity of the per-position arms rather than to its own ~64-token record
# contiguity, and roughly ten points of the difference is granularity, not signal.
WEIGHTED_SELECTS = LEXICAL_SELECTS + ("random-doc", "oracle-support")
# `oracle-support` is NOT an experimental arm and must never be reported as one: it
# reads support_document_ids, so it is an ORACLE. It exists to bound the question
# "how much of the uncompressed model's failure is fixable by better selection at
# all?" -- keep only the gold record and nothing else. If a blind selector already
# matches it, selection is saturated and the residual failures are the model's.
ORACLE_SELECTS = ("oracle-support",)
# Every arm whose selection consumes the RNG and therefore needs a per-case seed.
STOCHASTIC_SELECTS = ("random", "random-prepool") + WEIGHTED_SELECTS


# --------------------------------------------------------------------------- #
# 1. score recording
# --------------------------------------------------------------------------- #

def _score_rows(module, query: torch.Tensor, key: torch.Tensor,
                window: int, scaling: float | None) -> torch.Tensor:
    """Significance contribution of this forward pass. Returns [b, n_kv, n].

    `query` is [b, H, q, d] post-RoPE, `key` is [b, H_kv, n, d] post-RoPE and
    pre-`repeat_kv` -- exactly the grouping eviction needs.
    """
    b, h, q_len, d = query.shape
    n_kv, n = key.shape[1], key.shape[2]
    groups = h // n_kv
    w = min(window, q_len)
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(d)

    q = query[:, :, -w:, :].reshape(b, n_kv, groups, w, d).float()
    k = key.float()
    # [b, n_kv, groups, w, n]
    logits = torch.einsum("bgcwd,bgnd->bgcwn", q, k) * scaling

    if q_len > 1:
        # Observation-window rows sit at absolute positions n-w .. n-1.
        pos_q = torch.arange(n - w, n, device=query.device)
        pos_k = torch.arange(n, device=query.device)
        logits = logits.masked_fill(pos_k[None, :] > pos_q[:, None], float("-inf"))

    a = logits.softmax(-1)
    return a.sum(dim=(2, 3))  # over groups and window -> [b, n_kv, n]


def patch_attention() -> bool:
    """Wrap the SDPA attention interface so every call also records scores."""
    if _STATE["patched"]:
        return False
    from transformers.integrations import sdpa_attention as sa

    original = sa.sdpa_attention_forward

    def wrapped(module, query, key, value, attention_mask=None, dropout=0.0,
                scaling=None, is_causal=None, **kwargs):
        if _STATE["enabled"]:
            with torch.no_grad():
                s = _score_rows(module, query, key, _STATE["window"], scaling)
            li = getattr(module, "layer_idx", None)
            if li is not None:
                prev = _SCORES.get(li)
                pc = _COUNTS.get(li)
                ones = torch.ones_like(s)
                if prev is not None and prev.shape[-1] == s.shape[-1]:
                    _SCORES[li] = prev + s          # accumulate
                    _COUNTS[li] = (pc + ones) if (pc is not None and pc.shape == ones.shape) else ones
                elif prev is not None and prev.shape[-1] < s.shape[-1]:
                    # The cache grew; carry the accumulated history forward onto
                    # the prefix it covers.
                    pad = s.clone()
                    pad[..., : prev.shape[-1]] += prev
                    _SCORES[li] = pad
                    cpad = ones.clone()
                    if pc is not None:
                        cpad[..., : pc.shape[-1]] += pc
                    _COUNTS[li] = cpad
                elif prev is not None:
                    # prev is LONGER than s. Overwriting would throw away every
                    # observation accumulated so far and restart the ranking from
                    # one forward pass, which looks like a policy that simply
                    # prefers recent tokens. Unreachable today; reachable under
                    # chunked prefill and after a cache reset without
                    # reset_scores(). Fail rather than quietly forget.
                    raise RuntimeError(
                        f"layer {li}: accumulated scores cover {prev.shape[-1]} "
                        f"positions but this pass covers {s.shape[-1]}. Silently "
                        f"replacing them would discard all accumulated history. "
                        f"Call reset_scores() between sequences.")
                else:
                    _SCORES[li] = s
                    _COUNTS[li] = ones
        return original(module, query, key, value, attention_mask=attention_mask,
                        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)

    sa.sdpa_attention_forward = wrapped
    # The interface registry holds its own reference.
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
    except Exception:  # noqa: BLE001
        pass
    _STATE["patched"] = True
    return True


def set_recording(enabled: bool, window: int | None = None) -> None:
    _STATE["enabled"] = enabled
    if window is not None:
        _STATE["window"] = window


def reset_scores() -> None:
    _SCORES.clear()
    _COUNTS.clear()
    _LEXICAL.clear()
    _STATE["lexical_seed"] = None


def set_lexical_weights(weights: torch.Tensor | None) -> None:
    """Install the per-prompt-position trivial lexical weight for the next case.

    `weights` is 1-D, length = prompt length, values in [0, 1]; 0 means "this
    position's token id carries no lexical evidence about the question". It is
    broadcast to every KV head on the first eviction of each layer and then
    gathered through the surviving indices, so it stays aligned with the cache.

    Must be called per case, AFTER reset_scores(), by any caller using a
    `lexical-*` select. `select_keep` raises rather than falling back if it is
    missing -- a lexical arm that silently ranked by accumulated attention would
    write "lexical" into the manifest while measuring the mechanism, which is the
    same defect shape as `--select streaming` once yielding significance
    selection.
    """
    if weights is None:
        _STATE["lexical_seed"] = None
        return
    w = weights.detach().to(torch.float32).flatten()
    if w.numel() == 0:
        raise ValueError("set_lexical_weights: empty weight vector")
    if not torch.isfinite(w).all():
        raise ValueError("set_lexical_weights: non-finite weights")
    _STATE["lexical_seed"] = w


def _lexical_for(layer_idx: int, b: int, n_kv: int, n: int,
                 device) -> torch.Tensor:
    """[b, n_kv, n] lexical weight for this layer, materialised or lag-padded.

    Padding the decode lag with ZERO, not +inf: an un-scored newly generated
    token carries no lexical evidence, and it sits inside the protected recent
    window anyway. Padding with +inf would let the newest tokens win the heavy
    top-k unconditionally -- the failure `_evict` already guards for on the score
    path.
    """
    cur = _LEXICAL.get(layer_idx)
    if cur is None:
        seed = _STATE.get("lexical_seed")
        if seed is None:
            raise RuntimeError(
                f"layer {layer_idx}: a lexical select was requested but no "
                f"weights were installed. Call significance.set_lexical_weights() "
                f"per case after reset_scores(). Falling back to the accumulated "
                f"score here would measure the mechanism and label it the control.")
        cur = seed.to(device).view(1, 1, -1).expand(b, n_kv, -1).contiguous()
    have = cur.shape[-1]
    if have < n:
        pad = torch.zeros(cur.shape[0], cur.shape[1], n - have,
                          device=cur.device, dtype=cur.dtype)
        cur = torch.cat([cur, pad], dim=-1)
    elif have > n:
        raise RuntimeError(
            f"layer {layer_idx}: {have} lexical weights against a cache of {n}. "
            f"Truncating would misalign the mask; call reset_scores() between "
            f"sequences.")
    _LEXICAL[layer_idx] = cur
    return cur


def ranked_scores(layer_idx: int, exponent: float = 0.0):
    """Ranking key `sum / count**exponent`.

    The sum ratchets shut: a position that survived T compression rounds was
    scored T times and a newcomer once, so the heavy set freezes. The mean
    removes that advantage entirely, including the part of it that is real
    evidence. The exponent trades one bias against the other with one knob.

    MEASURED, 12 needles identical in every cell -- 128K, budget 2048, YaRN x4,
    query-aware, results/pk_128k_pe*.json. Accuracy at depths 0.1 / 0.5 / 0.9:

      e = 0.00 (sum)    0.58  0.08  0.00     mean 0.22
      e = 0.25          0.67  0.08  0.08     mean 0.28
      e = 0.50          0.67  0.00  0.33     mean 0.33
      e = 0.75          0.58  0.17  0.67     mean 0.47
      e = 1.00 (mean)   0.42  0.33  0.83     mean 0.53

    At depth 0.9 the raw sum retrieves NOTHING (0 of 12) and the mean retrieves
    10 of 12: exact McNemar +10/-0, p = 0.0020; within-needle permutation trend
    p < 0.0001. Pooled over all 36 needle-positions, 0.22 -> 0.53, p = 0.0074.

    e = 1 is the default and the interior is not worth tuning -- the mean column
    is monotone with no interior optimum. The cost is at the other end: at depth
    0.1 the mean is worse (0.58 -> 0.42), because long-accumulated early
    positions carry real evidence that normalising discards.

    An earlier version of this docstring reported a 6-trial version of this sweep
    with Cochran-Armitage and Fisher p-values. Those arms are paired on needles
    and both tests model independent groups, so the p-values were wrong and are
    withdrawn; two of their conclusions reversed under the paired re-run. See
    docs/FINDINGS_KV_PROGRAM.md section 4c.
    """
    s = _SCORES.get(layer_idx)
    if s is None:
        return None
    if exponent == 0.0:
        return s
    c = _COUNTS.get(layer_idx)
    if c is None or c.shape != s.shape:
        return s
    return s / c.clamp_min(1.0).pow(exponent)


def mean_scores(layer_idx: int):
    """Backwards-compatible alias for the exponent=1 endpoint."""
    return ranked_scores(layer_idx, 1.0)


def heavy_zero_count() -> int:
    """How many times a policy ran with significance selection disabled."""
    return _STATE.get("heavy_zero_count", 0)


def reset_heavy_zero_count() -> None:
    _STATE["heavy_zero_count"] = 0
    _STATE["heavy_zero_warned"] = False


def get_scores() -> dict[int, torch.Tensor]:
    return _SCORES


# --------------------------------------------------------------------------- #
# 2. cache
# --------------------------------------------------------------------------- #

class SignificanceLayer(DynamicLayer):
    """DynamicLayer that scores, evicts, and reports true position count.

    Eviction happens inside `update()` on purpose. The alternative -- prefill,
    stop, compress, then resume -- requires driving the decode loop by hand, and
    the previous experiment showed how easily a hand-rolled loop drifts from
    `model.generate` (23% of replies changed from a kernel difference alone).
    Self-compressing here keeps `generate` completely unmodified, so the eval is
    protocol-identical to the stored gate baselines by construction.
    """

    def __init__(self, layer_idx: int | None = None, policy=None, *a, **kw):
        super().__init__(*a, **kw)
        self.layer_idx = layer_idx
        self.policy = policy
        self.true_length = 0
        self.evictions = 0

    def update(self, key_states, value_states, *args, **kwargs):
        self.true_length += key_states.shape[-2]
        keys, values = super().update(key_states, value_states, *args, **kwargs)
        pol = self.policy
        if pol is not None and keys.shape[-2] > pol.budget + pol.slack:
            # Only evict once real scores exist for this layer.
            #
            # `update()` runs BEFORE the attention call that produces the scores,
            # so on the prefill pass there is nothing to select with. Evicting
            # here anyway falls back to sinks+recent -- which is precisely the
            # sliding window the previous experiment showed collapses this model,
            # and it would silently replace full prefill attention with it. The
            # first version of this file did exactly that: at a 512-token prompt
            # with budget 64, prefill attention saw 64 of 512 keys, prefill
            # timing looked 2.6x "faster" at 16K, and the rag_dev sweep
            # reproduced the sliding-window abstention collapse.
            #
            # Correct order: full prefill attention -> scores recorded -> the
            # first decode step's update compresses. That is SnapKV's ordering.
            if _SCORES.get(self.layer_idx) is not None:
                self._evict()
                keys, values = self.keys, self.values
        return keys, values

    def _evict(self) -> None:
        pol = self.policy
        k, v = self.keys, self.values
        b, n_kv, n, d = k.shape
        s = ranked_scores(self.layer_idx, pol.rank_exponent())

        if s is None:
            # Unreachable: update() only calls _evict once _SCORES has an entry.
            # It used to fall back to a zero vector, which makes select_keep
            # return sinks + recent -- a sliding window, silently, which is the
            # exact failure that invalidated an entire sweep once already.
            raise RuntimeError(
                f"layer {self.layer_idx}: _evict with no scores. This degrades to "
                f"sinks+recent (a sliding window) rather than failing, so it is "
                f"raised instead. Check the update() guard.")

        if s.shape[-1] != n:
            lag = n - s.shape[-1]
            n_recent = pol.n_recent()
            if lag < 0 or lag > n_recent:
                # Padding the lag with +inf is only safe while the un-scored
                # tokens all sit inside the protected recent window. Once the lag
                # exceeds it -- which is exactly what happens under chunked
                # prefill, where the lag is a whole chunk rather than one decode
                # token -- those +inf entries win the heavy top-k unconditionally
                # and the policy becomes sinks + newest-chunk. A sliding window
                # again, and again silently.
                raise RuntimeError(
                    f"layer {self.layer_idx}: score lag {lag} exceeds the recent "
                    f"window {n_recent} (cache {n}, scores {s.shape[-1]}). Padding "
                    f"this with +inf would silently degrade the policy to "
                    f"sinks+recent. Score the chunk before evicting it.")
            # Safe lag: the un-scored tokens are the newest and are protected by
            # the recent window regardless, so +inf only keeps them from being
            # chosen as victims.
            pad = torch.full((b, n_kv, lag), float("inf"),
                             device=k.device, dtype=torch.float32)
            s = torch.cat([s.to(k.device).float(), pad], dim=-1)

        lex = None
        if pol.select in WEIGHTED_SELECTS:
            lex = _lexical_for(self.layer_idx, b, n_kv, n, k.device)

        idx = select_keep(s, n, pol, k.device, lexical=lex)
        if lex is not None:
            # Same discipline as the scores and the counts: carry the mask
            # through the compaction or it addresses the wrong positions from the
            # second eviction onward.
            _LEXICAL[self.layer_idx] = lex.gather(2, idx).contiguous()
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        self.keys = k.gather(2, gather_idx).contiguous()
        self.values = v.gather(2, gather_idx).contiguous()
        # Keep the score vector aligned with the surviving positions, and drop the
        # +inf placeholders so they do not dominate the next selection.
        # Gather the RAW accumulated score, not the ranking key -- writing the
        # mean back would compound the division on every subsequent round.
        raw = _SCORES.get(self.layer_idx)
        if raw is not None and raw.shape[-1] == n:
            kept = raw.gather(2, idx)
        else:
            kept = s.gather(2, idx)
        _SCORES[self.layer_idx] = torch.where(
            torch.isfinite(kept), kept, torch.zeros_like(kept)).contiguous()
        _gather_counts(self.layer_idx, idx, n)
        self.evictions += 1

    def get_seq_length(self) -> int:
        # Deliberately NOT keys.shape[-2]: after eviction the stored length is
        # smaller than the number of positions consumed, and reporting the stored
        # length would silently re-index RoPE for every new token.
        return self.true_length

    def stored_length(self) -> int:
        if not self.is_initialized or self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]


def _gather_counts(layer_idx: int, idx: torch.Tensor, n: int) -> None:
    """Carry the observation counts through an eviction.

    This used to drop `_COUNTS` whenever its length did not match the cache, and
    that is the NORMAL case on the self-compressing path: `_evict` pads the score
    vector to cover the decode lag but the counts were left short, so the shape
    test failed on every eviction, the counts were popped, and `ranked_scores`
    silently fell back to the raw sum. `Policy.count_exponent` was therefore inert
    on that path -- a wrong ranking with no error, which is the failure mode this
    module exists to avoid.

    An un-scored position has been observed zero times, so the lag is padded with
    zeros; `ranked_scores` clamps the divisor to 1, giving score 0 / 1 = 0, which
    is correct for a position no query has looked at yet.
    """
    cnt = _COUNTS.get(layer_idx)
    if cnt is None:
        return
    have = cnt.shape[-1]
    if have < n:
        pad = torch.zeros(cnt.shape[0], cnt.shape[1], n - have,
                          device=cnt.device, dtype=cnt.dtype)
        cnt = torch.cat([cnt, pad], dim=-1)
    elif have > n:
        # More counts than cache. Dropping them would restart the ranking, so
        # fail rather than quietly resetting the exponent to zero.
        raise RuntimeError(
            f"layer {layer_idx}: {have} observation counts against a cache of "
            f"{n}. Silently dropping them makes count_exponent inert. Call "
            f"reset_scores() between sequences.")
    _COUNTS[layer_idx] = cnt.gather(2, idx.to(cnt.device)).contiguous()

def grow_spans(s: torch.Tensor, mult: float, max_reach: int) -> torch.Tensor:
    """Let each score flow forward through positions that are above the noise floor.

    WHY THIS EXISTS. P2.5 measured that a FIXED backward reach cannot serve variable
    span lengths: back=4 scores 1.000 on a 5-digit key and 0.000 on a 7-digit one,
    and at the law's own threshold (back = d-1) the margin narrows as spans grow --
    0.972 at d=4, 1.000 at d=5, 0.778 at d=7. Passkey lets you tune the reach because
    the span length is known. Natural language does not; spans there run 20-40 tokens
    and vary per instance. So derive the reach instead of tuning it.

    The score enters a span at its HEAD, because query-aware scoring rewards the
    query-shaped label introducing the payload (P2.2: framing survives in 54% of
    streams against 35% for the payload). Growing that head forward through
    positions that are themselves above the noise floor reproduces the span and
    stops where the score collapses to filler.

    HYSTERESIS NEEDS A SELECTIVE SEED AND A PERMISSIVE GATE, and getting that
    backwards is the error this docstring exists to prevent. An earlier version
    required a position to be in the top-k to be grown INTO. That is exactly wrong:
    a payload position is individually weak -- that is the whole problem -- so the
    top-k requirement excluded precisely the positions the mechanism is meant to
    rescue. It scored 0.00 at 16K/B=64 and emitted degenerate '77777777777'.

    Nor can the gate be a quantile. When signal is sparse the quantile lands inside
    the filler, the gate opens on everything, and growth floods the sequence so that
    the top-k after it is arbitrary. On a 200-position stream holding one 5-token
    span, a 0.95 quantile admitted all 200 positions.

    The gate is therefore relative to each stream's MEDIAN, which tracks the noise
    floor because filler dominates the distribution. `mult` is how far above it a
    position must stand to be grown into. It is scale-free, independent of sequence
    length and of how sparse the signal is, and it needs no prior knowledge of the
    head-to-payload ratio -- only that evidence outscores filler, which is what
    "evidence" means.
    """
    if mult is None or mult <= 0.0:
        return s
    finite = torch.where(torch.isinf(s), torch.full_like(s, float("-inf")), s)
    valid = torch.isfinite(finite)

    # THE MEDIAN MUST BE OVER VALID ENTRIES ONLY. This previously substituted 0.0
    # for every masked slot and took the median of the padded vector -- which is
    # not "each stream's median" at all. `select_keep` masks sinks and the recent
    # window to -inf BEFORE calling this, and at decode-step evictions n is only
    # budget + 1, so those masks are a large share of the vector. On the rag_dev
    # keep-0.25 geometry (budget 58, sinks 4, recent 31) they are 59% of it, the
    # median of the zero-padded vector is EXACTLY 0.0, and since scores are
    # non-negative softmax sums the gate `finite > 0 * mult` opens on every
    # eligible position: measured 24 of 24 admitted where the true median admits
    # 2. Growth then floods the stream and the top-k after it is arbitrary --
    # which is verbatim the failure the docstring above rejects a quantile for.
    # The implementation had been producing it by another route.
    #
    # Severity tracks the masked fraction, so this degrades every grow arm rather
    # than only the collapsed ones: at 31% masked (keep 0.50, and the passkey
    # grow cells) the floor still lands near the 27th percentile instead of the
    # 50th.
    masked_nan = torch.where(valid, finite, torch.full_like(finite, float("nan")))
    noise = masked_nan.nanmedian(dim=-1, keepdim=True).values
    # A row with no valid entries yields nan, and `finite > nan` is False
    # everywhere, so the gate closes and nothing is grown. That is the correct
    # degenerate behaviour: no eligible positions means no span to grow.
    gate = valid & (finite > noise * mult)
    grown = finite
    for _ in range(max(0, int(max_reach))):
        shifted = F.pad(grown[..., :-1], (1, 0), value=float("-inf"))
        nxt = torch.where(gate, torch.maximum(grown, shifted), grown)
        if torch.equal(nxt, grown):
            break
        grown = nxt
    return grown


def select_keep(s: torch.Tensor, n: int, policy, device,
                lexical: torch.Tensor | None = None) -> torch.Tensor:
    """Indices to keep: sinks + top-heavy + recent. Returns [b, n_kv, budget]."""
    b, n_kv = s.shape[0], s.shape[1]
    n_sink = min(policy.n_sinks(), n)
    n_recent = min(policy.n_recent(), max(0, n - n_sink))
    n_heavy = min(policy.heavy(), max(0, n - n_sink - n_recent))

    # Mask the always-kept regions FIRST, then pool.
    #
    # Doing it the other way round -- which this file did until now -- lets the
    # sinks bleed into the pooling window. R2 puts 46.3% of its attention mass on
    # the first 4 positions, so with pool=7 (padding 3) slots 4, 5 and 6 inherit
    # that mass through the max and win top-k unconditionally, on every prompt,
    # despite carrying no score of their own. Measured on a synthetic case with
    # heavy=8: pooling kept 1 of 4 genuine content tokens against 4 of 4 without
    # it, and spent 3 of the 8 heavy slots on positions 4-6.
    #
    # Masking first makes the pool window see -inf where the sinks were, so a
    # neighbour can only be pulled in by a real score.
    middle = s.clone()
    if n_sink:
        middle[..., :n_sink] = float("-inf")
    if n_recent:
        middle[..., n - n_recent:] = float("-inf")

    # THE TRIVIALITY CONTROL, substituted HERE -- after the masks, before the
    # pool. The question this arm answers is whether the accumulated-attention
    # score does anything a token-id overlap with the question could not, so the
    # lexical weight must pass through the SAME backward pool, the same top-k and
    # the same slot count as the score. Substituting it after the pool (which is
    # where `random` is substituted, below) would compare a pooled score against
    # an unpooled heuristic and attribute the pooling to the score.
    #
    # `random` is deliberately NOT moved to join it: its post-pool position is
    # what every committed random arm was measured with, and it is the cleaner
    # definition of the floor -- uniform among eligible positions, with no
    # neighbour correlation introduced by max-pooling a uniform field.
    # THE CONTIGUITY-MATCHED FLOOR. `random` is substituted AFTER the pool, so the
    # pool is inert for it and it keeps scattered singletons. Every lexical arm is
    # substituted BEFORE the pool, so its tie-break noise is max-pooled into
    # contiguous runs of pool_back+1 positions -- and contiguity alone preserves
    # spans. Measured on the campaign's own records with a CONSTANT weight vector
    # carrying zero lexical information: 0.097 (4K) / 0.042 (16K) gold-span
    # retention against post-pool random's 0.014 / 0.014. Small, but it is a
    # difference in PLUMBING, and R would charge it to the lexical signal. This arm
    # is that plumbing with the signal removed, and it is the floor R is computed
    # against. `random` is kept unchanged so the committed random arms stay
    # comparable.
    if policy.select == "random-prepool":
        r = torch.rand(middle.shape, device=middle.device, dtype=middle.dtype)
        middle = torch.where(torch.isfinite(middle), r,
                             torch.full_like(r, float("-inf")))

    if policy.select in WEIGHTED_SELECTS:
        if lexical is None:
            raise ValueError(
                f"select={policy.select!r} requires a lexical weight tensor; "
                f"select_keep was called without one. Ranking by the accumulated "
                f"score here would measure the mechanism and record it as the "
                f"control.")
        if lexical.shape != middle.shape:
            raise ValueError(
                f"lexical weights {tuple(lexical.shape)} do not match the score "
                f"field {tuple(middle.shape)}; a misaligned mask selects the "
                f"wrong positions silently.")
        # A tie-break three orders of magnitude below the smallest weight level,
        # drawn per KV head exactly as the `random` arm draws. Without it, topk
        # on a field of equal weights resolves ties by index and the arm becomes
        # "keep the earliest lexical matches", a positional policy wearing a
        # lexical label.
        jitter = torch.rand(middle.shape, device=middle.device,
                            dtype=middle.dtype) * 1e-3
        middle = torch.where(torch.isfinite(middle), lexical.to(middle.dtype) + jitter,
                             torch.full_like(middle, float("-inf")))

    # SnapKV pooling: an evidence span is useful as a span, and scoring each
    # position independently tends to keep scattered singletons out of several
    # different spans. Max-pooling over a width-p neighbourhood makes a
    # high-scoring token pull its neighbours in with it.
    # A pool_back/pool_fwd set while pool <= 1 would silently do nothing, which
    # is precisely the failure the pool=1 default already produced once in this
    # program -- a knob that looked set and was inert for every measurement.
    #
    # `grow` was NOT covered by this guard until 2026-08-29, and it is reached
    # through the same `pool > 1` branch: `Policy(pool=1, grow=2.0)` silently
    # disabled derived reach and reported a fixed-window run under a derived-reach
    # label. No committed result used that combination -- every result tree was
    # scanned before the guard was added, 0 of 205 config-bearing nodes hit it --
    # so nothing needs re-running. It is fixed because the next one would not have
    # been caught either: the guard named the failure class and then checked two of
    # the three knobs in it.
    if policy.pool <= 1 and (policy.pool_back or policy.pool_fwd
                             or policy.grow is not None):
        raise ValueError(
            "pool_back/pool_fwd/grow require pool > 1; pooling is disabled at "
            f"pool={policy.pool} and the window would be ignored "
            f"(pool_back={policy.pool_back}, pool_fwd={policy.pool_fwd}, "
            f"grow={policy.grow})")
    if policy.pool > 1 and n > policy.pool:
        import torch.nn.functional as F
        # The window is asymmetric in general. A centred window is the SnapKV
        # default and stays the default here, but it is not obviously the right
        # shape: P2.2 measured that pooling does not protect spans as such, it
        # propagates a high score to the NEIGHBOURS of a high scorer. Query-aware
        # scoring makes the query-like framing of an evidence span score high and
        # the payload score low -- 54% against 35% survival over 224 streams --
        # and the payload sits AFTER its label. Reaching further back than forward
        # therefore credits an answer for the label that introduces it. `back`
        # and `fwd` make that shape a parameter instead of an assumption.
        if policy.grow is not None:
            # Derived reach: grow each head through its own span (P2.4).
            pooled = grow_spans(middle, policy.grow, policy.grow_max)
        else:
            back, fwd = policy.pool_window()
            padded = F.pad(middle, (back, fwd), value=float("-inf"))
            pooled = F.max_pool1d(padded, kernel_size=back + fwd + 1,
                                  stride=1, padding=0)[..., :n]
        # Re-assert the masks: max_pool1d propagates finite values into the
        # masked regions, which would make sinks/recent eligible for the heavy
        # top-k and let a position be selected twice.
        if n_sink:
            pooled[..., :n_sink] = float("-inf")
        if n_recent:
            pooled[..., n - n_recent:] = float("-inf")
        middle = pooled

    # BASELINES. The mechanism has spent this whole programme being compared to
    # earlier versions of itself -- centred against causal, fixed reach against
    # derived. That establishes which variant is better and says nothing about
    # whether the significance score is doing any work at all. These two arms are
    # the missing controls.
    #
    #   random     keep the same NUMBER of positions, chosen uniformly among the
    #              eligible ones. Isolates the score from the budget: if random
    #              scores as well, the ranking is decorative. This control has
    #              already caught two false results in this programme.
    #   streaming  sinks + recent only, nothing ranked (heavy()==0, handled above).
    #              The standard cheap baseline; anything that cannot beat it is not
    #              worth its memory.
    if policy.select == "random" and n_heavy:
        r = torch.rand(middle.shape, device=middle.device, dtype=middle.dtype)
        middle = torch.where(torch.isfinite(middle), r,
                             torch.full_like(r, float("-inf")))

    parts = []
    if n_sink:
        parts.append(torch.arange(n_sink, device=device).view(1, 1, -1).expand(b, n_kv, -1))
    if n_heavy:
        parts.append(middle.topk(n_heavy, dim=-1).indices)
    if n_recent:
        parts.append(torch.arange(n - n_recent, n, device=device)
                     .view(1, 1, -1).expand(b, n_kv, -1))
    return torch.cat(parts, dim=-1).sort(dim=-1).values


class SignificanceCache(Cache):
    """Cache of SignificanceLayers.

    `DynamicCache.__init__` does not expose `layer_class_to_replicate` (it is a
    `Cache` argument), so the layer list is built explicitly from the config.
    """

    def __init__(self, config=None, policy=None, **kw):
        self.policy = policy
        if config is not None:
            n = config.get_text_config(decoder=True).num_hidden_layers
            layers = [SignificanceLayer(layer_idx=i, policy=policy) for i in range(n)]
            super().__init__(layers=layers, **kw)
        else:
            super().__init__(layer_class_to_replicate=SignificanceLayer, **kw)

    def stored_lengths(self) -> list[int]:
        return [l.stored_length() for l in self.layers]

    def total_evictions(self) -> int:
        return sum(getattr(l, "evictions", 0) for l in self.layers)


# --------------------------------------------------------------------------- #
# 3. eviction
# --------------------------------------------------------------------------- #

@dataclass
class Policy:
    budget: int = 128          # total positions kept per KV head, per layer
    sinks: int = 4             # always-kept prefix (46.3% of R2 attention mass)
    recent: int = 32           # always-kept suffix (persistence is only 0.584)
    protect_sinks: bool = True
    slack: int = 0             # evict only once stored exceeds budget + slack
    pool: int = 1              # max-pool width over scores before top-k (SnapKV)
    pool_back: int | None = None   # reach behind i; None = pool // 2 (centred)
    pool_fwd: int | None = None    # reach ahead of i;  None = pool // 2 (centred)
    select: str = "significance"   # "significance" | "random" | "streaming"
                                   # | "lexical-idf" | "lexical-naive".
                                   # All but the first are BASELINES, not policies
                                   # to deploy: they exist so the mechanism can be
                                   # compared against something other than earlier
                                   # versions of itself. See select_keep.
                                   #
                                   # The two lexical arms are the TRIVIALITY
                                   # CONTROL: they rank by whether a position's
                                   # token id occurs in the question, which needs
                                   # no forward pass at all. If they match the
                                   # score, the score is not the contribution.
                                   # They require set_lexical_weights() per case.
    grow: float | None = None      # if set, DERIVE the reach: geodesic dilation
                                   # gated at this MULTIPLE of the stream's median
                                   # score, instead of a fixed pool_back. 2.0 means
                                   # "twice the noise floor". See grow_spans.
    grow_max: int = 64             # iteration bound for growth
    # --- B1: the self-sizing budget -----------------------------------------
    # `recent` as a fixed constant is the defect that produced the largest wrong
    # conclusion in this programme. On a 232-token rag_dev prompt at keep 0.25
    # the whole budget is ~58, so sinks(4) + recent(32) took 61% of it and
    # heavy() hit zero on 17,360 compressions -- the policy silently became
    # StreamingLLM and we scored its absence as the mechanism underperforming.
    #
    # recent_frac makes the window scale with the budget. min_heavy_frac is the
    # part that matters: it RESERVES a share for the score, so the degenerate
    # case cannot occur at all rather than being counted after the fact. A
    # counter that reports "the mechanism did not run" is worth having, but not
    # having to.
    recent_frac: float | None = None   # recent as a fraction of budget; overrides `recent`
    min_heavy_frac: float = 0.0        # floor on the score's share of the budget
    normalize: bool = False    # shorthand for count_exponent = 1.0
    count_exponent: float | None = None   # rank on sum / count**e; 0 = sum, 1 = mean
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # An unrecognised `select` is the worst possible typo here: select_keep
        # special-cases known strings and otherwise ranks by the accumulated
        # score, so `select="lexical_idf"` (underscore) would run the MECHANISM
        # and write the control's name into the manifest. Same failure shape as
        # `--select streaming` once yielding significance selection, and as
        # pool_back-without-pool_fwd. Name the class, then close all of it.
        known = ("significance", "streaming") + STOCHASTIC_SELECTS
        if self.select not in known:
            raise ValueError(
                f"select={self.select!r} is not one of {known}. An unknown value "
                f"would silently fall through to significance ranking while the "
                f"manifest recorded the name you passed.")

    def rank_exponent(self) -> float:
        """Exponent actually used for ranking. `normalize` is the e=1 shorthand."""
        if self.count_exponent is not None:
            return float(self.count_exponent)
        return 1.0 if self.normalize else 0.0

    def pool_window(self) -> tuple[int, int]:
        """(reach behind, reach ahead) for the score pool.

        Defaults to the centred SnapKV window, so every result measured before
        this parameter existed reproduces exactly: pool=7 gives (3, 3), which is
        what max_pool1d(kernel=7, padding=3) did. Setting pool_back alone gives a
        causal window -- position i inherits only from positions at or before it,
        which is the shape that credits a payload for the label preceding it.
        """
        # Setting exactly one of the two is almost never what the caller means.
        # `--pool 7 --pool-back 6` silently yields (6, 3) -- fwd falls back to
        # pool//2 -- so a run intended to be causal measures a window that
        # reaches THREE POSITIONS INTO THE FUTURE. Every cornerstone-2 result
        # rests on fwd=0, and the failure is silent: the run completes, the
        # manifest looks plausible, and the number is of a different mechanism.
        # No committed run ever hit this (audited: 0 of them set one alone), so
        # raising costs nothing and closes it before it can happen.
        if (self.pool_back is None) != (self.pool_fwd is None):
            raise ValueError(
                f"pool_back={self.pool_back!r} and pool_fwd={self.pool_fwd!r}: set "
                f"BOTH or NEITHER. The unset one defaults to pool//2 = "
                f"{self.pool // 2}, so setting pool_back alone with pool="
                f"{self.pool} gives a window of ({self.pool_back}, "
                f"{self.pool // 2}) that reaches forward -- not the causal "
                f"window. Pass pool_fwd=0 explicitly for the cornerstone-2 shape.")
        half = self.pool // 2
        back = half if self.pool_back is None else int(self.pool_back)
        fwd = half if self.pool_fwd is None else int(self.pool_fwd)
        return back, fwd

    def n_sinks(self) -> int:
        return min(self.sinks if self.protect_sinks else 0, self.budget)

    def n_recent(self) -> int:
        # The always-kept regions must fit inside the budget, or the "budget" is
        # not a budget. sinks + recent = 36 silently overran a budget of 32.
        avail = max(0, self.budget - self.n_sinks())
        want = (self.recent if self.recent_frac is None
                else int(round(self.recent_frac * self.budget)))
        rec = min(max(0, int(want)), avail)

        # B1 floor. The recent window yields to the score, not the other way
        # round: `recent` is a heuristic for "positions too new to have a
        # trustworthy score", while the score is the mechanism under test. When
        # the two compete for a budget this small, starving the mechanism makes
        # the run measure nothing. The `streaming` baseline is exempt -- it sets
        # heavy to zero deliberately, and that is its whole purpose.
        if self.min_heavy_frac > 0.0 and self.select == "significance":
            reserve = max(1, int(math.ceil(self.min_heavy_frac * self.budget)))
            rec = min(rec, max(0, avail - reserve))
            # THE RECENT WINDOW IS NOT PURELY A HEURISTIC -- it also absorbs the
            # SCORE LAG. Between a token entering the cache and being scored
            # there is at least one un-scored position, and `_evict` raises if
            # that lag exceeds n_recent, because padding it with +inf would let
            # the newest tokens win the heavy top-k unconditionally and silently
            # turn the policy into sinks + newest-chunk.
            #
            # The first version of this floor drove recent to 0 at every budget
            # <= 20. rag_dev prompts start at 41 tokens, so keep 0.25 produces
            # budgets from ~10, and all three B1 arms died on the first short
            # prompt. The 24 CPU tests passed throughout: they asserted heavy > 0
            # across budgets 5..1024 and never asserted recent >= 1, because they
            # tested Policy in isolation and never ran the eviction path. That is
            # the same gap that let grow_spans flood its gate under eleven green
            # tests -- a unit test that does not reproduce the caller's
            # preconditions pins the wrong function.
            if avail >= 1:
                rec = max(1, rec)
        return rec

    def heavy(self) -> int:
        """Slots chosen by significance. Zero means the policy is StreamingLLM.

        The `streaming` baseline sets this to zero deliberately, so the budget goes
        entirely to sinks plus a recent window and nothing is ranked at all.

        Deliberately warns rather than raising: a raise mid-sweep truncates the
        run and leaves a partial results.jsonl that the summarizer will happily
        aggregate. The count is exported so a run manifest can record how many
        items silently had no significance selection at all -- on rag_dev, 20 of
        752 prompts hit this at every keep fraction.
        """
        h = max(0, self.budget - self.n_sinks() - self.n_recent())
        if h == 0:
            _STATE["heavy_zero_count"] = _STATE.get("heavy_zero_count", 0) + 1
            if not _STATE.get("heavy_zero_warned"):
                _STATE["heavy_zero_warned"] = True
                import warnings
                warnings.warn(
                    f"Policy.heavy() == 0 (budget {self.budget}, sinks "
                    f"{self.n_sinks()}, recent {self.n_recent()}): significance "
                    f"selection is disabled and this is plain StreamingLLM. "
                    f"Budget must exceed sinks + recent. Further occurrences are "
                    f"counted in significance.heavy_zero_count(), not warned.",
                    RuntimeWarning, stacklevel=2)
        return h


def compress(cache, policy: Policy, scores: dict[int, torch.Tensor] | None = None) -> dict:
    """Evict from every layer down to `policy.budget` positions per KV head.

    Returns a small report so callers can log what was dropped rather than
    assume it worked.
    """
    scores = scores if scores is not None else _SCORES
    report = {"layers": 0, "before": 0, "after": 0, "skipped": 0}

    for li, layer in enumerate(cache.layers):
        if not getattr(layer, "is_initialized", False) or layer.keys is None:
            continue
        k, v = layer.keys, layer.values
        b, n_kv, n, d = k.shape
        report["before"] += n
        if n <= policy.budget or li not in scores:
            report["after"] += n
            report["skipped"] += 1
            continue

        s = scores[li].to(k.device).float()          # [b, n_kv, n]
        e = policy.rank_exponent()
        if e != 0.0:
            m = ranked_scores(li, e)
            if m is not None and m.shape == s.shape:
                s = m.to(k.device).float()
        if s.shape[-1] != n:
            # Skipping here leaves that layer's cache FULL while the report says
            # the pass succeeded, so a run reads as "compression applied" with
            # some layers uncompressed -- an invisible, layer-dependent budget.
            raise RuntimeError(
                f"layer {li}: scores cover {s.shape[-1]} positions but the cache "
                f"holds {n}. compress() used to skip this layer and leave it "
                f"uncompressed while reporting success.")

        idx = select_keep(s, n, policy, k.device)             # [b, n_kv, B]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        layer.keys = k.gather(2, gather_idx).contiguous()
        layer.values = v.gather(2, gather_idx).contiguous()
        raw = _SCORES.get(li)
        scores[li] = ((raw if raw is not None and raw.shape[-1] == n else s)
                      .gather(2, idx).contiguous())
        _gather_counts(li, idx, n)

        report["layers"] += 1
        report["after"] += idx.shape[-1]

    report["reduction"] = (report["before"] / report["after"]) if report["after"] else 1.0
    return report


def cache_bytes(cache) -> int:
    total = 0
    for layer in getattr(cache, "layers", []):
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def score_bytes(scores: dict[int, torch.Tensor] | None = None) -> int:
    """Bytes held by the SCORE tensors alone.

    This is NOT the mechanism's full overhead. `_COUNTS` is a second tensor of
    identical shape and dtype, and count normalisation -- the part that makes the
    ranking work at long context -- cannot run without it. Reporting this number
    as "the significance matrix costs X" understates it by exactly 2x, which is
    what happened: 0.78% was published where the mechanism as used costs 1.56%.
    Use `matrix_bytes()` for the whole thing.
    """
    scores = scores if scores is not None else _SCORES
    return sum(t.numel() * t.element_size() for t in scores.values())


def matrix_bytes() -> int:
    """Bytes held by the whole significance matrix: scores AND observation counts."""
    return (sum(t.numel() * t.element_size() for t in _SCORES.values())
            + sum(t.numel() * t.element_size() for t in _COUNTS.values()))
