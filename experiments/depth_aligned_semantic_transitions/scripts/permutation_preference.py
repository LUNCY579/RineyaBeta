"""Does the UNTOUCHED model prefer some state-to-depth orderings over others?

Two questions, both answered without training anything:

Q1. THE MATCHED FROZEN FLOOR. The findings report quotes a trained order margin of
    0.935 as evidence the mechanism was learned. But a fitted linear probe on the
    FROZEN backbone reached 0.977-0.995 in Phase 0. If that holds under the same
    geometry, then the depth trajectory was already linearly decodable before any
    training, and a high trained margin is not evidence that training created it.

Q2. PERMUTATION PREFERENCE. For 4 stages there are 4! = 24 assignments of semantic
    states to depth checkpoints. Fit a shared linear readout for each, score on
    held-out problems, and rank them. If the untouched network systematically
    prefers reverse-ish permutations over the canonical forward one, that is
    evidence of an intrinsic semantic ordering of depth -- and it would explain the
    trained L_state asymmetry (reverse 0.253 vs ordered 0.263) without invoking
    "transformers reason backwards".

CPU only. Uses the cached pooled hidden states so no model forward passes are
needed for the representations; the frozen embedding is loaded only to build targets.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import dast.tokenization as _dt

_o = _dt._as_int_list


def _p(v):
    if hasattr(v, "keys") and "input_ids" in v:
        v = v["input_ids"]
    return _o(v)


_dt._as_int_list = _p

from dast.config import ARM_SPECS, load_config  # noqa: E402
from dast.data import build_examples, load_records  # noqa: E402
from dast.semantic import FrozenTextTargetEncoder  # noqa: E402

DEV, CFG, CACHE, RESULT = (Path(sys.argv[1]), Path(sys.argv[2]),
                           Path(sys.argv[3]), sys.argv[4])
cfg = load_config(CFG)
K = cfg.num_stages

blob = torch.load(CACHE)
H = blob["SYM27"].float()          # the corrected readout: final-norm at 7/14/21/27
N = H.shape[0]
print(f"cached hidden states: {tuple(H.shape)}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path, local_files_only=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg.model.name_or_path, local_files_only=True, dtype=torch.float32).eval()
for q in model.parameters():
    q.requires_grad_(False)
enc = FrozenTextTargetEncoder(
    model.get_input_embeddings(), tok, state_dim=cfg.semantic.state_dim,
    projection_seed=cfg.semantic.target_projection_seed,
    max_tokens=cfg.semantic.target_max_tokens)

records = load_records(DEV, split="dev", limit=N)
examples = build_examples(records, arm=ARM_SPECS["state_ordered"],
                          num_stages=K, seed=cfg.experiment.seed)
S = enc.encode_grid([list(e.trajectory.canonical_texts) for e in examples], device="cpu")
# The shipped protocol centres targets within problem.
S = F.normalize(S - S.mean(dim=1, keepdim=True), dim=-1)
print(f"targets: {tuple(S.shape)} (within-problem centred, as the runs use)")

half = N // 2
eye = torch.eye(K, dtype=torch.bool)


def fit_and_score(targets: torch.Tensor, lam: float = 1.0) -> dict:
    """One SHARED linear readout across depths, fitted on half, scored on the rest."""

    Xtr = torch.cat([H[:half].reshape(-1, H.shape[-1]), torch.ones(half * K, 1)], 1)
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1]),
                           Xtr.T @ targets[:half].reshape(-1, targets.shape[-1]))
    Xte = H[half:]
    flat = torch.cat([Xte.reshape(-1, H.shape[-1]), torch.ones((N - half) * K, 1)], 1)
    Z = F.normalize((flat @ W).reshape(N - half, K, -1), dim=-1)
    Yte = F.normalize(targets[half:], dim=-1)
    C = torch.einsum("nkd,njd->nkj", Z, Yte)
    diag = C[:, eye].mean().item()
    return {
        "L_state": 1.0 - diag,                      # what the training objective minimises
        "diagonal_cosine": diag,
        "order_margin": diag - C[:, ~eye].mean().item(),
        "top1": (C.argmax(2) == torch.arange(K)).float().mean().item(),
    }


out: dict = {"records": N, "state_dim": cfg.semantic.state_dim, "readout": "final_norm 7/14/21/27"}

# --- Q1: the matched frozen floor, canonical (identity) assignment -------------
out["Q1_frozen_floor_identity"] = fit_and_score(S)

# --- Q2: all 24 permutations --------------------------------------------------
rows = []
identity = tuple(range(K))
reverse = tuple(reversed(range(K)))
for perm in itertools.permutations(range(K)):
    # depth k is asked to predict canonical state perm[k]
    scored = fit_and_score(S[:, list(perm), :])
    rows.append({
        "perm": list(perm),
        "is_identity": perm == identity,
        "is_reverse": perm == reverse,
        "kendall_tau_vs_identity": sum(
            1 if (perm[i] < perm[j]) else -1
            for i in range(K) for j in range(i + 1, K)
        ) / (K * (K - 1) / 2),
        **scored,
    })
rows.sort(key=lambda r: r["L_state"])
out["Q2_permutations_best_first"] = rows

ident = next(r for r in rows if r["is_identity"])
rev = next(r for r in rows if r["is_reverse"])
out["summary"] = {
    "identity_rank": rows.index(ident) + 1,
    "reverse_rank": rows.index(rev) + 1,
    "total_permutations": len(rows),
    "identity_L_state": ident["L_state"],
    "reverse_L_state": rev["L_state"],
    "best_L_state": rows[0]["L_state"],
    "worst_L_state": rows[-1]["L_state"],
    "best_perm": rows[0]["perm"],
    "worst_perm": rows[-1]["perm"],
}
Path(RESULT).write_text(json.dumps(out, indent=2), encoding="utf-8")

print()
print("=== Q1: matched frozen floor, canonical assignment, final metric ===")
for k, v in out["Q1_frozen_floor_identity"].items():
    print(f"   {k:20s} {v:.4f}")
print()
print("=== Q2: all 24 permutations, best L_state first ===")
print(f"   {'rank':>4} {'perm':>14} {'L_state':>9} {'margin':>9} {'top1':>7}  tau   note")
for index, r in enumerate(rows, start=1):
    note = "IDENTITY (forward)" if r["is_identity"] else ("REVERSE" if r["is_reverse"] else "")
    print(f"   {index:>4} {str(r['perm']):>14} {r['L_state']:>9.4f} "
          f"{r['order_margin']:>9.4f} {r['top1']:>7.4f} {r['kendall_tau_vs_identity']:>+5.2f}  {note}")
print()
print(f"identity rank {out['summary']['identity_rank']}/24, "
      f"reverse rank {out['summary']['reverse_rank']}/24")
print(f"wrote {RESULT}")
