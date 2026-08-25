#!/usr/bin/env python3
"""Per-position trace: gate signals, deletion damage and remnant benefit.

This is the raw material for two separate questions the audit keeps apart:

1. **Is the confidence signal a good proxy for recovery need?** (brief section 6)
   For that we need, per token, a confidence measurement and the *actual* benefit
   of running the remnant -- not a downstream accuracy number, which conflates
   the gate's skill with everything else in the pipeline.

2. **What should the probe be trained on?** The historical probe predicts model
   confidence. The preregistration for SGQR-v2 says a probe should instead
   predict *recovery utility*. Both are fitted here from the same trace so the
   comparison is like-for-like.

Three forwards per prompt: dense, all-four-deleted, all-four-remnants-on. The
gate signal is read from the residual entering the FIRST retired layer, which no
arm modifies, so one capture serves every arm.

Writes a JSONL with one row per reply position. Nothing here is aggregated; the
analysis scripts do that, so a re-analysis never needs the GPU again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (CALIB, TEST, VAL, RETIRED, SPECS, N_GROUPS, REPO,        # noqa: E402
                  env_manifest, install, lens_features, load_model,
                  load_split, make_head, reply_logprobs, ResidualTap)

import scripts.train.memory as mem                                        # noqa: E402
from envinfo import require_cuda                                          # noqa: E402

SPLITS = {"calib": CALIB, "val": VAL, "test": TEST}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calib", choices=sorted(SPLITS))
    ap.add_argument("--bits", default="int4", choices=sorted(SPECS))
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test on the first N prompts of the split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(20260824)

    model, tok = load_model(device)
    prompts = load_split(tok, SPLITS[args.split], max_len=args.max_len)
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        raise SystemExit("no usable prompts")
    head = make_head(model)
    inst = install(model, spec=SPECS[args.bits])
    first = min(RETIRED)
    tap = ResidualTap(model, first)

    out_dir = REPO / "experiments/audit_sgqr_csgqr/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) if args.out else (
        out_dir / ("trace_%s_%s.jsonl" % (args.split, args.bits)))
    dest = dest if dest.is_absolute() else REPO / dest

    print("trace  split=%s (%d prompts)  bits=%s  retired=%s"
          % (args.split, len(prompts), args.bits, list(RETIRED)))
    t0 = time.time()
    n_rows = 0
    hidden_chunks: list[torch.Tensor] = []
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        for pi, p in enumerate(prompts):
            ids = p["ids"].to(device)
            b = p["boundary"]
            gold = ids[0, b:].detach().cpu()

            inst.use_dense()
            dense_lp = reply_logprobs(model, ids, b)
            h = tap.buf[0, b - 1:-1].detach()          # reply-predicting rows
            feats = lens_features(head, h)

            inst.use_remnant()
            inst.set_allocation_by_level([{l: [] for l in RETIRED}])
            inst.set_levels(None)
            del_lp = reply_logprobs(model, ids, b)

            inst.set_allocation_by_level([{l: list(range(N_GROUPS)) for l in RETIRED}])
            rem_lp = reply_logprobs(model, ids, b)
            inst.use_dense()

            pd = dense_lp.exp()
            kl_del = (pd * (dense_lp - del_lp)).sum(dim=-1).cpu()
            kl_rem = (pd * (dense_lp - rem_lp)).sum(dim=-1).cpu()
            d_arg = dense_lp.argmax(dim=-1).cpu()
            x_arg = del_lp.argmax(dim=-1).cpu()
            r_arg = rem_lp.argmax(dim=-1).cpu()
            nll_d = -dense_lp.gather(1, gold.to(device).reshape(-1, 1)).squeeze(1).cpu()
            nll_x = -del_lp.gather(1, gold.to(device).reshape(-1, 1)).squeeze(1).cpu()
            nll_r = -rem_lp.gather(1, gold.to(device).reshape(-1, 1)).squeeze(1).cpu()

            hidden_chunks.append(h.float().cpu())
            for t in range(kl_del.numel()):
                fh.write(json.dumps({
                    "prompt": pi, "pos": t, "prompt_id": p.get("id"),
                    "gold": int(gold[t]),
                    "dense_top1": int(d_arg[t]), "deleted_top1": int(x_arg[t]),
                    "remnant_top1": int(r_arg[t]),
                    "kl_deleted": float(kl_del[t]), "kl_remnant": float(kl_rem[t]),
                    "utility": float(kl_del[t] - kl_rem[t]),
                    "nll_dense": float(nll_d[t]), "nll_deleted": float(nll_x[t]),
                    "nll_remnant": float(nll_r[t]),
                    "max_prob": float(feats["max_prob"][t]),
                    "neg_entropy": float(feats["neg_entropy"][t]),
                    "prob_margin": float(feats["prob_margin"][t]),
                    "logit_margin": float(feats["logit_margin"][t]),
                }) + "\n")
                n_rows += 1
            if (pi + 1) % 16 == 0:
                print("  %d/%d  %.0fs" % (pi + 1, len(prompts), time.time() - t0),
                      flush=True)

    tap.close()
    hid = torch.cat(hidden_chunks, dim=0)
    torch.save({"hidden": hid, "split": args.split, "bits": args.bits,
                "layer": first, "n_rows": n_rows},
               dest.with_suffix(".hidden.pt"))
    manifest = {
        "split": args.split, "range": list(SPLITS[args.split]),
        "prompts": len(prompts), "positions": n_rows, "bits": args.bits,
        "retired_layers": list(RETIRED), "n_groups": N_GROUPS,
        "signal_layer": first, "max_len": args.max_len,
        "hidden_dim": int(hid.shape[1]), "elapsed_s": round(time.time() - t0, 1),
        "environment": env_manifest(),
    }
    dest.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("wrote %s  (%d rows, %.0fs)" % (dest.name, n_rows, time.time() - t0))


if __name__ == "__main__":
    main()
