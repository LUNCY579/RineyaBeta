"""QRR-1: bounded query-conditioned re-reading over an explicit evidence bank.

Tensor prototype only. Caller must supply CAUSALLY computed hidden states; masking
cannot undo future information already mixed into those states by an encoder.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .identity import ContractError, positive_int


def norm(x):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class ReadStep(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.query = nn.Linear(rank, rank, bias=False)
        self.update = nn.Linear(2 * rank, rank, bias=False)
        self.gate = nn.Linear(2 * rank, rank, bias=False)

    def forward(self, state, keys, values, mask):
        q = self.query(norm(state))
        scores = torch.einsum("br,bnr->bn", q.float(), keys.float()) / keys.shape[-1] ** .5
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
        read = torch.einsum("bn,bnr->br", weights, values.float()).to(state.dtype)
        pair = torch.cat((norm(state), read), dim=-1)
        revised = state + .25 * torch.sigmoid(self.gate(pair)) * torch.tanh(self.update(pair))
        return revised, weights


@dataclass
class ReadResult:
    residual: torch.Tensor
    attention: tuple
    state_changes: tuple


class QueryReadRevise(nn.Module):
    def __init__(self, hidden=1024, rank=64, reads=2, tied=True):
        super().__init__()
        positive_int(hidden, "hidden", 65536)
        positive_int(rank, "rank", 1024)
        if reads not in (1, 2, 4) or type(reads) is not int or type(tied) is not bool:
            raise ContractError("Reads must be 1/2/4; tied must be boolean")
        self.hidden, self.rank, self.reads, self.tied = hidden, rank, reads, tied
        self.input = nn.Linear(hidden, rank, bias=False)
        self.key = nn.Linear(hidden, rank, bias=False)
        self.value = nn.Linear(hidden, rank, bias=False)
        self.output = nn.Linear(rank, hidden, bias=False)
        self.steps = nn.ModuleList(ReadStep(rank) for _ in range(1 if tied else reads))
        # One zero-initialized output projection, NOT a second closed scalar gate.
        # Its gradient is live first; controller gradients open after this moves.
        nn.init.zeros_(self.output.weight)

    def forward(self, query, evidence, visible, query_positions, key_positions, trace=False):
        if query.ndim != 2 or evidence.ndim != 3:
            raise ContractError("Expected query[B,H] and evidence[B,N,H]")
        b, n, h = evidence.shape
        if h != self.hidden or query.shape != (b, h) or n == 0:
            raise ContractError("Invalid hidden/batch/evidence shape")
        if visible.dtype != torch.bool or visible.shape != (b, n):
            raise ContractError("Explicit boolean evidence visibility required")
        if query_positions.shape != (b,) or key_positions.shape != (b, n):
            raise ContractError("Absolute position shapes do not match")
        if query_positions.dtype not in (torch.int32, torch.int64) or key_positions.dtype not in (torch.int32, torch.int64):
            raise ContractError("Positions must be integer tensors")
        values = (evidence, visible, query_positions, key_positions)
        if any(t.device != query.device for t in values) or evidence.dtype != query.dtype:
            raise ContractError("Tensors must share device and hidden dtype")
        if not query.is_floating_point() or not bool(torch.isfinite(query).all()):
            raise ContractError("Query must be finite floating point")
        mask = visible & (key_positions <= query_positions[:, None]) & (key_positions >= 0)
        if not bool(mask.any(dim=-1).all()):
            raise ContractError("Every query needs at least one legal evidence token")
        # Remove inaccessible padding/future values before projections, including NaNs.
        clean = evidence.masked_fill(~mask[..., None], 0)
        if not bool(torch.isfinite(clean).all()):
            raise ContractError("Visible evidence must be finite")
        keys, vals = self.key(norm(clean)), self.value(norm(clean))
        state = self.input(norm(query))
        attention, changes = [], []
        for i in range(self.reads):
            previous = state
            state, weights = self.steps[0 if self.tied else i](state, keys, vals, mask)
            if trace:
                attention.append(weights.detach())
                changes.append((state - previous).float().norm(dim=-1).detach())
        residual = self.output(norm(state))
        if not bool(torch.isfinite(residual).all()):
            raise ContractError("Non-finite workspace output")
        return ReadResult(residual, tuple(attention), tuple(changes))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class MatchedMLP(nn.Module):
    """Nearest fully active two-matrix MLP; exact at registered H=1024,R=64."""
    def __init__(self, hidden, target_parameters):
        super().__init__()
        self.width = max(1, round(target_parameters / (2 * hidden)))
        self.input = nn.Linear(hidden, self.width, bias=False)
        self.output = nn.Linear(self.width, hidden, bias=False)
        nn.init.zeros_(self.output.weight)
        self.count = sum(p.numel() for p in self.parameters())
        self.target_parameters = target_parameters

    def forward(self, query):
        return self.output(torch.nn.functional.silu(self.input(norm(query))))


def add_logit_residual(native_logits, residual, head):
    """Explicit integration seam, not an installed HF hook or cache implementation."""
    if not isinstance(head, nn.Linear) or head.bias is not None:
        raise ContractError("Only a verified bias-free linear LM head is supported")
    if native_logits.shape[:-1] != residual.shape[:-1] or residual.shape[-1] != head.in_features:
        raise ContractError("Residual/logit shapes mismatch")
    if native_logits.shape[-1] != head.out_features:
        raise ContractError("LM-head vocabulary mismatch")
    return native_logits + head(residual).to(native_logits.dtype)


def arm_table(hidden=1024, rank=64):
    rows = []
    for reads, tied in ((1, True), (2, True), (4, True), (2, False)):
        model = QueryReadRevise(hidden, rank, reads, tied)
        count = model.parameter_count()
        expected = 4 * hidden * rank + 5 * rank * rank * (1 if tied else reads)
        if count != expected:
            raise ContractError("Analytical parameter counter disagrees")
        rows.append({"arm": f"qrr_{reads}_{'tied' if tied else 'untied'}", "parameters": count,
                     "reads": reads, "native_qualified": False})
    mlp = MatchedMLP(hidden, rows[0]["parameters"])
    rows.append({"arm": "matched_mlp", "parameters": mlp.count,
                 "exact_parameter_match": mlp.count == mlp.target_parameters,
                 "extra_inactive_padding_parameters": 0, "native_qualified": False})
    return {"evidence_class": "TENSOR_PROTOTYPE_ONLY", "arms": rows,
            "compute_matched": False, "latency_measured": False,
            "warning": "Equal stored parameters do not imply equal training or inference work"}


def initialized_controls(hidden=1024, rank=64, seed=17):
    """Matched initial shared tensors; untied cells begin as exact shared copies."""
    import copy
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        one = QueryReadRevise(hidden, rank, 1, True)
        two, four, untied = (copy.deepcopy(one) for _ in range(3))
        two.reads, four.reads, untied.reads, untied.tied = 2, 4, 2, False
        untied.steps.append(copy.deepcopy(one.steps[0]))
        mlp = MatchedMLP(hidden, one.parameter_count())
    return {"one": one, "two": two, "four": four, "untied_two": untied, "mlp": mlp}
