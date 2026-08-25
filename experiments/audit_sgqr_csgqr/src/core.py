#!/usr/bin/env python3
"""Shared core for the SGQR / CSGQR dual audit.

Design rule
-----------
SGQR and CSGQR must differ by **configuration only**. In the historical code they
do not: ``multilayer.SGQRMlp`` runs one fixed group set with no per-token branch,
while ``surgery.CSGQRMlp`` buckets every token through a Python loop. Any latency
or even quality comparison between those two modules confounds the confidence
gate with two different execution engines -- and CSGQR-v1 measured that engine
difference at 16x wall-clock.

So this audit builds ONE module, :class:`LevelledSGQRMlp`, which executes a
per-position *level* index. A fixed-budget SGQR arm is that module with every
position on the same level; a confidence-gated CSGQR arm is the same module with
levels assigned by a gate. Same code, same kernels, same allocator behaviour.
The only difference between the two arms is the integer vector of levels.

Matched compute is exact, not approximate
-----------------------------------------
Every arm at a budget point is given the **same total number of group
executions**, to the unit, by :func:`allocate_exact`. The historical campaign
matched activation only to ~0.5% (0.6772 against 0.6819) and then compared KL,
which leaves a compute difference inside the comparison. Here the total is
identical by construction and the per-arm realized activation is asserted equal.

Splits
------
``chat_dev`` has 510 usable prompts at ``max_len=256``. The historical campaign
used 0..95 for calibration and 96..509 as one held-out set, and then read
exploratory quantities (the probe/lens crossing point) off that held-out set.
This audit uses three disjoint splits so that nothing is selected on the set it
is reported on:

    calib  prompts   0..95    (96)  significance ranking, probe fitting
    val    prompts  96..223  (128)  every threshold, every hyperparameter
    test   prompts 224..509  (286)  confirmatory, reported once
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CSGQR_SRC = REPO / "experiments" / "csgqr" / "src"
V2_SRC = REPO / "experiments" / "sgqr_v2" / "src"
for _p in (str(HERE), str(V2_SRC), str(CSGQR_SRC), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envinfo import collect, require_cuda                            # noqa: E402
from measure_layers import DATASETS, load_prompts                    # noqa: E402
from quant import QuantSpec                                          # noqa: E402
from multilayer import (RetirementPlan, install_sgqr,                # noqa: E402
                        magnitude_scores)

MODEL = "artifacts/rineya-core-en32k-28L"
RETIRED = (13, 14, 24, 25)
N_GROUPS = 24
TOTAL_GROUPS = len(RETIRED) * N_GROUPS          # 96

CALIB = (0, 96)
VAL = (96, 224)
TEST = (224, 510)

SPECS = {"int4": QuantSpec(4, 128, False), "int8": QuantSpec(8, 128, False),
         "fp16": QuantSpec(16, 0, True)}


# ==========================================================================
# Execution engine
# ==========================================================================
class LevelledSGQRMlp(nn.Module):
    """One retired MLP, executing a per-position group set chosen by level.

    ``groups_by_level[i]`` is the group list this layer runs for any position
    assigned level ``i``. ``set_levels`` installs the per-position assignment for
    the next forward. With a single level this is exactly ``SGQRMlp``; with more
    it is a gated arm. Inactive groups are sliced away before dequantization, so
    a level with fewer groups genuinely executes less work.
    """

    def __init__(self, remnant, groups_by_level: list[list[int]]) -> None:
        super().__init__()
        self.remnant = remnant
        self.groups_by_level = [sorted(set(int(g) for g in lvl))
                                for lvl in groups_by_level]
        for lvl in self.groups_by_level:
            if lvl and not (0 <= lvl[0] and lvl[-1] < remnant.n_groups):
                raise ValueError("group id outside [0, %d)" % remnant.n_groups)
        self.levels: Tensor | None = None
        self.executed_groups = 0
        self.executed_tokens = 0

    def set_groups_by_level(self, groups_by_level: list[list[int]]) -> None:
        self.groups_by_level = [sorted(set(int(g) for g in lvl))
                                for lvl in groups_by_level]

    def set_levels(self, levels: Tensor | None) -> None:
        self.levels = None if levels is None else levels.reshape(-1).to(torch.long)

    def reset_counters(self) -> None:
        self.executed_groups = 0
        self.executed_tokens = 0

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        n = flat.shape[0]
        out = torch.zeros_like(flat)
        if self.levels is None:
            lv = torch.zeros(n, dtype=torch.long)
        elif self.levels.numel() == n:
            lv = self.levels
        else:
            raise RuntimeError("level vector has %d entries for %d positions; "
                               "a mismatched assignment would silently route "
                               "the wrong tokens" % (self.levels.numel(), n))
        self.executed_tokens += n
        for i in torch.unique(lv).tolist():
            groups = self.groups_by_level[i]
            rows = (lv == i).nonzero(as_tuple=True)[0]
            if not groups or rows.numel() == 0:
                continue
            self.executed_groups += len(groups) * int(rows.numel())
            idx = rows.to(flat.device)
            out.index_copy_(0, idx,
                            self.remnant.forward_groups(flat.index_select(0, idx),
                                                        groups))
        return out.reshape(shape)


@dataclass
class Installed:
    model: nn.Module
    modules: dict[int, LevelledSGQRMlp]
    pristine: dict[int, nn.Module]
    plan: RetirementPlan
    inner: object

    def use_dense(self) -> None:
        for l, m in self.pristine.items():
            self.model.model.layers[l].mlp = m

    def use_remnant(self) -> None:
        for l, m in self.modules.items():
            self.model.model.layers[l].mlp = m

    def set_allocation_by_level(self, alloc_by_level: list[dict[int, list[int]]]) -> None:
        """``alloc_by_level[i][layer]`` = groups layer runs at level ``i``."""
        for l, m in self.modules.items():
            m.set_groups_by_level([a.get(l, []) for a in alloc_by_level])

    def set_levels(self, levels: Tensor | None) -> None:
        for m in self.modules.values():
            m.set_levels(levels)

    def reset_counters(self) -> None:
        for m in self.modules.values():
            m.reset_counters()

    def executed_groups(self) -> int:
        return sum(m.executed_groups for m in self.modules.values())

    def executed_tokens(self) -> int:
        ms = list(self.modules.values())
        return ms[0].executed_tokens if ms else 0

    def packed_bytes(self) -> int:
        return sum(m.remnant.packed_bytes()["total"] for m in self.modules.values())

    def dense_bytes(self) -> int:
        return sum(m.remnant.dense_bytes() for m in self.modules.values())


def install(model, *, spec: QuantSpec, donor_map: dict[int, int] | None = None,
            layers: tuple[int, ...] = RETIRED,
            n_groups: int = N_GROUPS) -> Installed:
    """Retire ``layers`` with a levelled remnant each, donors kept resident."""
    plan = RetirementPlan(layers=layers, spec=spec, n_groups=n_groups,
                          donor_map=donor_map)
    inner = install_sgqr(model, plan, allocation={l: [] for l in layers},
                         keep_donor_resident=True)
    modules, pristine = {}, {}
    for l in layers:
        rem = inner.modules[l].remnant
        modules[l] = LevelledSGQRMlp(rem, [list(range(n_groups))])
        pristine[l] = inner.originals[l]
        model.model.layers[l].mlp = modules[l]
    return Installed(model=model, modules=modules, pristine=pristine, plan=plan,
                     inner=inner)


# ==========================================================================
# Budget allocation
# ==========================================================================
def global_order(scores: dict[int, list[float]],
                 layers: tuple[int, ...] = RETIRED) -> list[tuple[int, int]]:
    """(layer, group) pairs, most useful first, ties by (layer, group)."""
    flat = [(float(scores[l][g]), l, g) for l in layers for g in range(len(scores[l]))]
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(l, g) for _, l, g in flat]


def alloc_from_order(order: list[tuple[int, int]], k: int,
                     layers: tuple[int, ...] = RETIRED) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {l: [] for l in layers}
    for l, g in order[:max(0, int(k))]:
        out[l].append(g)
    return {l: sorted(v) for l, v in out.items()}


def random_order(seed: int, layers: tuple[int, ...] = RETIRED,
                 n_groups: int = N_GROUPS) -> list[tuple[int, int]]:
    total = len(layers) * n_groups
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=gen).tolist()
    return [(layers[i // n_groups], i % n_groups) for i in perm]


def allocate_exact(weights: Tensor, total_budget: int, *,
                   max_k: int = TOTAL_GROUPS, tie_seed: int = 20260824) -> Tensor:
    """Per-position integer budgets summing to EXACTLY ``total_budget``.

    ``weights`` are non-negative per-position demands (higher = more remnant
    wanted). The allocation is proportional to ``weights`` and rounded by the
    largest-remainder (Hamilton) method, which is deterministic and hits the
    total exactly. A flat ``weights`` vector therefore produces the fixed-budget
    arm and a confidence-derived one produces the gated arm, with **identical**
    total compute -- which is what "matched compute" has to mean if the
    comparison is to be about allocation rather than about spend.
    """
    w = weights.to(torch.float64).clamp(min=0.0)
    n = w.numel()
    total_budget = int(total_budget)
    if total_budget <= 0:
        return torch.zeros(n, dtype=torch.long)
    if total_budget >= n * max_k:
        return torch.full((n,), max_k, dtype=torch.long)
    s = float(w.sum())
    if s <= 0:
        w = torch.ones(n, dtype=torch.float64)
        s = float(n)
    ideal = w * (total_budget / s)
    # cap at max_k, redistributing the overflow onto the uncapped positions
    for _ in range(64):
        over = ideal > max_k
        if not bool(over.any()):
            break
        excess = float((ideal[over] - max_k).sum())
        ideal = torch.where(over, torch.full_like(ideal, float(max_k)), ideal)
        free = ~over
        fs = float(ideal[free].sum())
        if fs <= 0 or not bool(free.any()):
            break
        ideal = torch.where(free, ideal * (1.0 + excess / fs), ideal)
    base = torch.floor(ideal).to(torch.long).clamp(0, max_k)
    remainder = total_budget - int(base.sum())
    # Ties in the fractional part are the rule, not the exception: a FLAT demand
    # vector makes every fraction identical. Breaking those ties by position
    # index would hand every spare group to the first positions of the first
    # prompts, so the "uniform" arm would not be uniform. A fixed permutation
    # spreads them deterministically instead.
    tb = torch.randperm(n, generator=torch.Generator().manual_seed(tie_seed))
    if remainder > 0:
        frac = ideal - base.to(torch.float64)
        frac = torch.where(base >= max_k, torch.full_like(frac, -1.0), frac)
        order = sorted(range(n), key=lambda i: (-float(frac[i]), int(tb[i])))
        for i in order[:remainder]:
            base[i] += 1
    elif remainder < 0:
        frac = ideal - base.to(torch.float64)
        frac = torch.where(base <= 0, torch.full_like(frac, 2.0), frac)
        order = sorted(range(n), key=lambda i: (float(frac[i]), int(tb[i])))
        for i in order[:-remainder]:
            base[i] -= 1
    assert int(base.sum()) == total_budget, (int(base.sum()), total_budget)
    return base


def binary_budgets(need: Tensor, total_budget: int, *,
                   max_k: int = TOTAL_GROUPS) -> Tensor:
    """Full remnant on the neediest tokens, nothing on the rest, exact total.

    This is the binary gate: the whole remnant either runs for a token or does
    not. Exactness costs one partial token at the boundary, which is recorded
    rather than rounded away.
    """
    n = need.numel()
    total_budget = int(total_budget)
    k = torch.zeros(n, dtype=torch.long)
    n_full = min(n, total_budget // max_k)
    rest = total_budget - n_full * max_k
    order = sorted(range(n), key=lambda i: (-float(need[i]), i))
    for i in order[:n_full]:
        k[i] = max_k
    if rest > 0 and n_full < n:
        k[order[n_full]] = rest
    assert int(k.sum()) == total_budget
    return k


def levels_from_budgets(k: Tensor, ladder: list[int]) -> Tensor:
    """Map exact per-position budgets onto the nearest ladder rung.

    Used only for reporting; execution uses the exact budgets via a per-prompt
    level table, so no rounding touches the compute accounting.
    """
    lad = torch.tensor(ladder, dtype=torch.long)
    return (k.reshape(-1, 1) - lad.reshape(1, -1)).abs().argmin(dim=1)


# ==========================================================================
# Data
# ==========================================================================
def load_split(tokenizer, split: tuple[int, int], dataset: str = "chat_dev",
               max_len: int = 256) -> list[dict]:
    lo, hi = split
    prompts = load_prompts(REPO / DATASETS[dataset], tokenizer, hi, max_len, skip=0)
    return prompts[lo:hi]


def load_model(device: str = "cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False
    return model, tok


# ==========================================================================
# Forward helpers
# ==========================================================================
@torch.no_grad()
def reply_logprobs(model, ids: Tensor, boundary: int) -> Tensor:
    return torch.log_softmax(model(ids).logits[0, boundary - 1:-1].float(), dim=-1)


class ResidualTap:
    """Captures the pre-MLP residual entering a layer.

    The residual entering the FIRST retired layer is produced entirely by layers
    below it plus that layer's own attention, none of which any arm modifies. It
    is therefore **arm-invariant**: capturing it once per prompt from the dense
    pass gives every arm literally the same gate signal, which removes an entire
    class of confound from the gate comparison.
    """

    def __init__(self, model, layer: int) -> None:
        self.buf: Tensor | None = None
        self.handle = model.model.layers[layer].post_attention_layernorm \
            .register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        self.buf = args[0].detach()
        return None

    def close(self) -> None:
        self.handle.remove()


@torch.no_grad()
def lens_features(head, h: Tensor, chunk: int = 256) -> dict[str, Tensor]:
    """max_prob / neg_entropy / prob_margin / logit_margin from a residual.

    Chunked over positions so a [T, |V|] tensor never exists for a whole prompt
    (CSGQR-v1 bug S8). Vocabulary tensors die inside this function.
    """
    outs = {k: [] for k in ("max_prob", "neg_entropy", "prob_margin", "logit_margin")}
    flat = h.reshape(-1, h.shape[-1])
    for i in range(0, flat.shape[0], chunk):
        z = head(flat[i:i + chunk]).float()
        lp = torch.log_softmax(z, dim=-1)
        p = lp.exp()
        top2 = z.topk(2, dim=-1).values
        ptop2 = p.topk(2, dim=-1).values
        outs["max_prob"].append(ptop2[:, 0])
        outs["neg_entropy"].append((p * lp).sum(dim=-1))
        outs["prob_margin"].append(ptop2[:, 0] - ptop2[:, 1])
        outs["logit_margin"].append(top2[:, 0] - top2[:, 1])
        del z, lp, p, top2, ptop2
    return {k: torch.cat(v).cpu() for k, v in outs.items()}


def make_head(model):
    norm = model.model.norm
    lm = model.lm_head

    @torch.no_grad()
    def head(h: Tensor) -> Tensor:
        return lm(norm(h))
    return head


# ==========================================================================
# Metrics
# ==========================================================================
def position_metrics(dense_lp: Tensor, arm_lp: Tensor, gold: Tensor) -> dict[str, Tensor]:
    p = dense_lp.exp()
    d_top1 = dense_lp.argmax(dim=-1)
    a_top1 = arm_lp.argmax(dim=-1)
    idx = gold.reshape(-1, 1)
    return {
        "kl": (p * (dense_lp - arm_lp)).sum(dim=-1),
        "nll_gold": -arm_lp.gather(1, idx).squeeze(1),
        "agree": (a_top1 == d_top1).to(torch.float32),
        "gold_top1": (a_top1 == gold).to(torch.float32),
    }


def env_manifest(extra: dict | None = None) -> dict:
    out = collect()
    if extra:
        out.update(extra)
    return out
