#!/usr/bin/env python3
"""Does each arm keep the SYSTEM PROMPT? The generator for the instruction-retention
table in docs/FINDINGS_TRIVIALITY_CONTROL.md section 4.

WHY THIS EXISTS AS A COMMITTED SCRIPT. The first version of that table came from an
ad-hoc scratchpad simulation that used ONE KV head and ONE eviction, and it was never
committed -- so its numbers could not be reproduced or checked, and one of them
(a refusal-span figure) turned out to be unreproducible at any setting. A number that
supports a published finding has to come from a script in the repository.

WHAT MAKES THIS THE REAL PATH, not a reimplementation:
  * the weight vectors come from run_eval_sig.py's own doc_bm25_weights /
    doc_random_weights / lexical_weights -- the functions the campaign ran;
  * selection is scripts.kv.significance.select_keep itself, over the real 8 KV
    heads, through the real SignificanceLayer.update eviction sequence, with the
    mask carried by the real _LEXICAL gather;
  * keys are tagged with their original position index in channel 0, so survivors
    are read off the cache rather than inferred.

No model weights and no GPU are needed, because the trivial arms' selection is
provably independent of the accumulated-attention score: driving select_keep with
zeros, U(0,1e4) and arange gives byte-identical selections for every `lexical-*`,
`random-doc` and `random-prepool` arm. The significance arm IS score-dependent and is
therefore NOT simulated here -- its row comes from the real runs (see --check).

Usage:
    python scripts/design/instruction_survival.py            # write
    python scripts/design/instruction_survival.py --check    # verify, non-zero on drift
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/kv_significance/results/instruction_survival.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_res", ROOT / "experiments/kv_significance/src/run_eval_sig.py")
_res = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_res)

DATASET = ROOT / "artifacts/eval_sets/rag_long4k_d0.5.jsonl"
LIMIT = 200
KEEP_FRAC, SINKS, RECENT_FRAC = 0.25, 4, 0.15
N_KV, DECODE_STEPS = 8, 20          # the core's KV heads; replies run ~10-25 tokens
SEED = 20260826
REFUSAL = "I don't know based on the available information."

ARMS = [
    ("G", "lexical-doc-bm25", "trivial: document-level BM25"),
    ("D", "lexical-idf", "trivial: per-position IDF"),
    ("E", "lexical-naive", "trivial: naive question-token match"),
    ("Ndoc", "random-doc", "floor: granularity-matched random"),
    ("N", "random-prepool", "floor: contiguity-matched random"),
]


def weights_for(select: str, prompt: str, enc, question: str, tok) -> torch.Tensor:
    n = len(enc["input_ids"])
    if select == "lexical-doc-bm25":
        return _res.doc_bm25_weights(prompt, enc, question, tok)
    if select == "random-doc":
        return _res.doc_random_weights(prompt, enc)
    if select == "random-prepool":
        return torch.zeros(n)          # no weights: the arm is pure pooled noise
    return _res.lexical_weights(enc["input_ids"], question, tok, select)


def run_case(select: str, w: torch.Tensor, n: int, sys_end: int,
             refusal: list[int] | None) -> dict:
    """Drive the real eviction path; return survival at the 1st and last eviction."""
    budget = max(1, int(round(KEEP_FRAC * n)))
    pol = sig.Policy(budget=budget, sinks=SINKS, recent_frac=RECENT_FRAC,
                     pool=7, pool_back=6, pool_fwd=0, select=select)
    sig.reset_scores()
    if select in sig.WEIGHTED_SELECTS:
        sig.set_lexical_weights(w)
    layer = sig.SignificanceLayer(layer_idx=0, policy=pol)
    k = torch.zeros(1, N_KV, n, 2)
    k[..., 0] = torch.arange(n, dtype=torch.float32)     # channel 0 = original index
    layer.update(k, k.clone())
    sig._SCORES[0] = torch.zeros(1, N_KV, n)             # inert for the trivial arms
    first, last = None, None
    for step in range(DECODE_STEPS):
        t = torch.zeros(1, N_KV, 1, 2)
        t[..., 0] = -1.0                                 # generated tokens: not prompt
        layer.update(t, t.clone())
        surv = layer.keys[..., 0]                        # [1, N_KV, budget]
        frac = float(((surv >= 0) & (surv < sys_end)).float().sum(-1).mean()
                     / max(sys_end, 1))
        intact = 0.0
        if refusal:
            want = set(float(x) for x in refusal)
            per_head = [len(want & set(surv[0, h].tolist())) == len(want)
                        for h in range(N_KV)]
            intact = sum(per_head) / N_KV
        if first is None:
            first = (frac, intact)
        last = (frac, intact)
    return {"sys_frac_first": first[0], "refusal_intact_first": first[1],
            "sys_frac_last": last[0], "refusal_intact_last": last[1]}


def derive() -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts/rineya-core-en32k-28L"))
    recs = [json.loads(l) for l in
            DATASET.read_text(encoding="utf-8").splitlines() if l.strip()][:LIMIT]
    acc = {a: {kk: [] for kk in ("sys_frac_first", "refusal_intact_first",
                                 "sys_frac_last", "refusal_intact_last")}
           for a, _, _ in ARMS}
    sys_lens, n_ref = [], 0
    for i, rec in enumerate(recs):
        msgs = (rec["messages"][:-1] if rec["messages"][-1]["role"] == "assistant"
                else rec["messages"])
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        enc = tok(prompt, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        n = len(ids)
        umark = prompt.find("<|im_start|>user")
        sys_end = next((j for j, (a, z) in enumerate(offs) if a >= umark), n)
        sys_lens.append(sys_end)
        rstart = prompt.find(REFUSAL)
        refusal = ([j for j, (a, z) in enumerate(offs)
                    if a >= rstart and z <= rstart + len(REFUSAL)]
                   if rstart >= 0 else None)
        if refusal:
            n_ref += 1
        for arm, select, _lab in ARMS:
            torch.manual_seed(SEED + i)
            w = weights_for(select, prompt, enc, rec["question"], tok)
            r = run_case(select, w, n, sys_end, refusal)
            for kk, vv in r.items():
                acc[arm][kk].append(vv)
    rows = {}
    for arm, select, label in ARMS:
        rows[arm] = {"select": select, "label": label,
                     **{kk: round(sum(v) / len(v), 4) for kk, v in acc[arm].items()}}
    return {"dataset": "rag_long4k", "n_cases": len(recs),
            "keep_frac": KEEP_FRAC, "sinks": SINKS, "recent_frac": RECENT_FRAC,
            "n_kv_heads": N_KV, "decode_steps": DECODE_STEPS, "seed": SEED,
            "mean_system_prompt_tokens": round(sum(sys_lens) / len(sys_lens), 1),
            "cases_containing_the_refusal_text": n_ref,
            "note": ("Fractions are means over cases AND over the 8 KV heads. "
                     "'first' is after the first eviction (when the answer's opening "
                     "tokens are produced); 'last' after decode_steps evictions. The "
                     "significance arm is NOT here: its selection depends on the "
                     "accumulated score, which needs a real forward pass."),
            "arms": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    d = derive()
    print(f"system prompt: {d['mean_system_prompt_tokens']} tokens mean; "
          f"refusal text present in {d['cases_containing_the_refusal_text']}/"
          f"{d['n_cases']} prompts; {d['n_kv_heads']} KV heads; "
          f"{d['decode_steps']} evictions")
    print(f"\n{'arm':6s} {'select':18s} {'sys kept (1st)':>15s} {'(last)':>9s} "
          f"{'refusal intact (1st)':>21s} {'(last)':>9s}")
    for a, r in d["arms"].items():
        print(f"{a:6s} {r['select']:18s} {r['sys_frac_first']:15.4f} "
              f"{r['sys_frac_last']:9.4f} {r['refusal_intact_first']:21.4f} "
              f"{r['refusal_intact_last']:9.4f}")
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"no committed artifact at {OUT}")
        if json.loads(OUT.read_text(encoding="utf-8")) != d:
            raise SystemExit("DRIFT: recomputed survival differs from the committed one")
        print("\ncheck: committed instruction-survival table reproduces")
        return
    OUT.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
