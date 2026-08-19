#!/usr/bin/env python3
"""Gate C3/C4: prove the vocabulary trim changed nothing it should not have.

The trim removes embedding rows and renumbers ids. It does not touch any weight
value. So for any text whose tokenization is unchanged by the trim, the claim is
strong and checkable:

    trimmed_logits[:, new_id]  ==  stock_logits[:, old_id]   for every kept id

Small floating-point differences are expected and reported rather than assumed
away: the lm_head GEMM changes shape from 151,936 to 32,768 output columns, so
BLAS may tile and accumulate the reduction differently. What must hold exactly is
the *ranking* -- argmax and top-5 agreement -- because that is what generation
consumes.

Also reports pre-heal generation quality (gate C4). If the trim is sound this
should be indistinguishable from stock, not merely recoverable.

Usage: python verify_trim.py <trimmed_model_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
STOCK = ROOT / "Qwen3-0.6B"
MAX_NEW = 64

PROMPTS = (
    "If Ivan is taller than Sara and Sara is taller than Mo, who is shortest?",
    "Give the shortest possible answer: what is 17 times 3, minus 5?",
    "The user said their meeting moved to Friday. Earlier they said Tuesday. "
    "Which day is the meeting?",
    "Evidence: the warehouse closes at 6pm on weekdays. Question: can I collect "
    "a parcel at 7pm on a Tuesday?",
    "Reply naturally and briefly: I just finished a really long week at work.",
)


def chat(tok, text: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main() -> None:
    trimmed_dir = Path(sys.argv[1])
    manifest = json.loads((trimmed_dir / "trim_manifest.json").read_text(encoding="utf-8"))
    mapping = {int(k): v for k, v in manifest["old_to_new_id"].items()}

    stock_tok = AutoTokenizer.from_pretrained(STOCK)
    trim_tok = AutoTokenizer.from_pretrained(trimmed_dir)

    texts = [chat(stock_tok, p) for p in PROMPTS]
    stock_ids = [stock_tok(t, return_tensors="pt").input_ids for t in texts]
    trim_ids = [trim_tok(chat(trim_tok, p), return_tensors="pt").input_ids
                for p in PROMPTS]

    print("--- tokenization agreement ---")
    comparable = []
    for i, (a, b) in enumerate(zip(stock_ids, trim_ids)):
        remapped = [mapping.get(int(t), -1) for t in a[0]]
        same = len(remapped) == b.shape[1] and remapped == [int(t) for t in b[0]]
        print(f"  prompt {i}: stock {a.shape[1]:>3d} tok, trimmed {b.shape[1]:>3d} tok"
              f"  -> {'identical after remap' if same else 'DIFFERENT (inflated)'}")
        if same:
            comparable.append(i)
    if not comparable:
        raise SystemExit("no prompt tokenizes identically; cannot run the equivalence check")

    print("\nloading stock ...")
    stock = AutoModelForCausalLM.from_pretrained(STOCK, dtype=torch.float32).eval()
    stock_logits, stock_gen = {}, {}
    with torch.no_grad():
        for i in comparable:
            stock_logits[i] = stock(stock_ids[i]).logits[0, -1].clone()
        for i, ids in enumerate(stock_ids):
            out = stock.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
            stock_gen[i] = stock_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    del stock

    print("loading trimmed ...")
    trim = AutoModelForCausalLM.from_pretrained(trimmed_dir, dtype=torch.float32).eval()
    print(f"  vocab_size {trim.config.vocab_size:,d}   "
          f"params {sum(p.numel() for p in trim.parameters()):,d}   "
          f"tied {trim.config.tie_word_embeddings}")

    print("\n--- logit equivalence on kept ids ---")
    print(f"{'prompt':>7s} {'max|diff|':>11s} {'argmax':>8s} {'top5':>7s} {'exact rows':>11s}")
    with torch.no_grad():
        for i in comparable:
            tl = trim(trim_ids[i]).logits[0, -1]
            sl = stock_logits[i]
            gather = torch.tensor([mapping[o] for o in sorted(mapping)], dtype=torch.long)
            olds = torch.tensor(sorted(mapping), dtype=torch.long)
            a, b = sl.index_select(0, olds), tl.index_select(0, gather)
            diff = (a - b).abs()
            am = mapping.get(int(sl.argmax())) == int(tl.argmax())
            t5s = {mapping.get(int(x)) for x in sl.topk(5).indices}
            t5t = {int(x) for x in tl.topk(5).indices}
            print(f"{i:>7d} {diff.max().item():>11.3e} {'yes' if am else 'NO':>8s} "
                  f"{'yes' if t5s == t5t else 'NO':>7s} "
                  f"{int((diff == 0).sum()):>7,d}/{len(olds):,d}")

    print("\n--- pre-heal generation (gate C4), greedy, thinking off ---")
    identical = 0
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = trim_ids[i]
            out = trim.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
            text = trim_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            match = text.strip() == stock_gen[i].strip()
            identical += match
            print(f"\n[{i}] {prompt[:66]}")
            print(f"  stock  : {stock_gen[i].strip()[:180]!r}")
            print(f"  trimmed: {text.strip()[:180]!r}")
            print(f"  identical to stock: {'yes' if match else 'no'}")

    print(f"\n--- verdict ---")
    print(f"generations identical to stock: {identical}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
