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
_STATE = {"enabled": False, "window": 16, "patched": False}


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


def ranked_scores(layer_idx: int, exponent: float = 0.0):
    """Ranking key `sum / count**exponent`.

    The two endpoints were measured and fail in OPPOSITE directions at 128K:

      exponent 0.0 (raw sum)   favours long-accumulated EARLY positions.
                               128K depth 0.1/0.5/0.9 -> 1.00 / 0.33 / 0.00
      exponent 1.0 (mean)      over-corrects toward LATE arrivals.
                               128K depth 0.1/0.5/0.9 -> 0.17 / 0.17 / 0.83

    The sum ratchets shut because a position that survived T compression rounds
    was scored T times and a newcomer once; the mean removes that advantage
    entirely, including the part of it that is real evidence. An exponent between
    the two trades one bias against the other with a single knob.
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

        idx = select_keep(s, n, pol, k.device)
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
        cnt = _COUNTS.get(self.layer_idx)
        if cnt is not None and cnt.shape[-1] == n:
            _COUNTS[self.layer_idx] = cnt.gather(2, idx).contiguous()
        else:
            _COUNTS.pop(self.layer_idx, None)
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


def select_keep(s: torch.Tensor, n: int, policy, device) -> torch.Tensor:
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

    # SnapKV pooling: an evidence span is useful as a span, and scoring each
    # position independently tends to keep scattered singletons out of several
    # different spans. Max-pooling over a width-p neighbourhood makes a
    # high-scoring token pull its neighbours in with it.
    if policy.pool > 1 and n > policy.pool:
        import torch.nn.functional as F
        pooled = F.max_pool1d(middle, kernel_size=policy.pool, stride=1,
                              padding=policy.pool // 2)[..., :n]
        # Re-assert the masks: max_pool1d propagates finite values into the
        # masked regions, which would make sinks/recent eligible for the heavy
        # top-k and let a position be selected twice.
        if n_sink:
            pooled[..., :n_sink] = float("-inf")
        if n_recent:
            pooled[..., n - n_recent:] = float("-inf")
        middle = pooled

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
    normalize: bool = False    # shorthand for count_exponent = 1.0
    count_exponent: float | None = None   # rank on sum / count**e; 0 = sum, 1 = mean
    stats: dict = field(default_factory=dict)

    def rank_exponent(self) -> float:
        """Exponent actually used for ranking. `normalize` is the e=1 shorthand."""
        if self.count_exponent is not None:
            return float(self.count_exponent)
        return 1.0 if self.normalize else 0.0

    def n_sinks(self) -> int:
        return min(self.sinks if self.protect_sinks else 0, self.budget)

    def n_recent(self) -> int:
        # The always-kept regions must fit inside the budget, or the "budget" is
        # not a budget. sinks + recent = 36 silently overran a budget of 32.
        return min(self.recent, max(0, self.budget - self.n_sinks()))

    def heavy(self) -> int:
        """Slots chosen by significance. Zero means the policy is StreamingLLM.

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
        cnt = _COUNTS.get(li)
        if cnt is not None and cnt.shape[-1] == n:
            _COUNTS[li] = cnt.gather(2, idx).contiguous()
        else:
            _COUNTS.pop(li, None)

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
    scores = scores if scores is not None else _SCORES
    return sum(t.numel() * t.element_size() for t in scores.values())
