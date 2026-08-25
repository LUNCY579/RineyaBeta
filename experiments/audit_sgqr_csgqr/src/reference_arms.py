#!/usr/bin/env python3
"""Reference arms on the audit's own test split: bitwidth ladder and donor identity.

These are the arms the brief lists as required context for the SGQR/CSGQR
comparison -- dense, hard deletion, the full quantized remnant at each bitwidth,
and the layer-identity controls -- re-measured on the audit's test slice
(prompts 224-509) so that every number in this report comes from one split and
one code path.

The donor-identity controls are three distinct wrong mappings rather than one, so
"the wrong donor is worse" is a claim about donor identity and not about one
unlucky pair of layers:

    shift1   layer j gets layer j+1's weights   (adjacent, most similar)
    shift6   layer j gets layer j+6's weights
    rev      the four retired layers' donors are permuted among themselves,
             reversed, so the *set* of installed weights is unchanged and only
             the assignment moves. This is the sharpest identity control
             available: same bytes, same weights, different placement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (N_GROUPS, REPO, RETIRED, SPECS, TEST, VAL, CALIB,     # noqa: E402
                  env_manifest, install, load_model, load_split,
                  position_metrics)
from stats import (cluster_bootstrap_mean,                              # noqa: E402
                   cluster_bootstrap_mean_pair)

import scripts.train.memory as mem                                      # noqa: E402
from envinfo import require_cuda                                        # noqa: E402

RES = REPO / "experiments/audit_sgqr_csgqr/results"
SPLITS = {"calib": CALIB, "val": VAL, "test": TEST}


def donor_maps(n_layers: int) -> dict[str, dict[int, int] | None]:
    rev = {l: d for l, d in zip(RETIRED, tuple(reversed(RETIRED)))}
    return {
        "correct": None,
        "shift1": {l: (l + 1) % n_layers for l in RETIRED},
        "shift6": {l: (l + 6) % n_layers for l in RETIRED},
        "rev-among-retired": rev,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=sorted(SPLITS))
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cap-mib", type=float, default=5120.0)
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(20260824)

    model, tok = load_model(device)
    prompts = load_split(tok, SPLITS[args.split], max_len=args.max_len)
    n_layers = model.config.num_hidden_layers
    DM = donor_maps(n_layers)

    print("reference arms on split=%s (%d prompts)" % (args.split, len(prompts)))
    dense_ref: list[torch.Tensor] = []
    gold_all: list[torch.Tensor] = []
    clusters: list[int] = []
    t0 = time.time()
    for pi, p in enumerate(prompts):
        ids = p["ids"].to(device)
        b = p["boundary"]
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0, b - 1:-1].float(), dim=-1)
        dense_ref.append(lp.cpu())
        gold_all.append(ids[0, b:].cpu())
        clusters.extend([pi] * int(lp.shape[0]))
    print("  dense reference cached in %.0fs (%d positions)"
          % (time.time() - t0, len(clusters)))

    ARMS = [("A-dense", None, None),
            ("B-delete4", "int4", "correct"),          # empty allocation
            ("C-fp16-full", "fp16", "correct"),
            ("C-int8-full", "int8", "correct"),
            ("C-int4-full", "int4", "correct"),
            ("G-shift1-int4", "int4", "shift1"),
            ("G-shift6-int4", "int4", "shift6"),
            ("G-rev-int4", "int4", "rev-among-retired")]

    out_arms: dict[str, dict] = {}
    raw: dict[str, dict[str, torch.Tensor]] = {}
    for name, bits, dm in ARMS:
        t1 = time.time()
        kls, nlls, agrs, glds, flips = [], [], [], [], []
        inst = None
        if bits is not None:
            inst = install(model, spec=SPECS[bits], donor_map=DM[dm])
            groups = [] if name == "B-delete4" else list(range(N_GROUPS))
            inst.set_allocation_by_level([{l: groups for l in RETIRED}])
            inst.set_levels(None)
        for pi, p in enumerate(prompts):
            ids = p["ids"].to(device)
            b = p["boundary"]
            if inst is not None:
                inst.use_remnant()
            with torch.no_grad():
                lp = torch.log_softmax(model(ids).logits[0, b - 1:-1].float(), dim=-1)
            if inst is not None:
                inst.use_dense()
            m = position_metrics(dense_ref[pi].to(device), lp, gold_all[pi].to(device))
            kls.append(m["kl"].cpu())
            nlls.append(m["nll_gold"].cpu())
            agrs.append(m["agree"].cpu())
            glds.append(m["gold_top1"].cpu())
            flips.append(float((m["agree"] < 0.5).any().item()))
            del lp, m
        if inst is not None:
            inst.use_dense()
            for l in RETIRED:
                model.model.layers[l].mlp = inst.pristine[l]
            packed = inst.packed_bytes()
            dense_b = inst.dense_bytes()
            del inst
            torch.cuda.empty_cache()
        else:
            packed = dense_b = None
        kl = torch.cat(kls).double()
        nll = torch.cat(nlls).double()
        gld = torch.cat(glds).double()
        raw[name] = {"kl": kl, "nll": nll, "gold": gld,
                     "agree": torch.cat(agrs).double()}
        out_arms[name] = {
            "arm": name, "bits": bits, "donor_map": dm,
            "n_positions": int(kl.numel()), "n_prompts": len(prompts),
            "kl": cluster_bootstrap_mean(kl, clusters, n_boot=args.n_boot),
            "gold_top1": cluster_bootstrap_mean(gld, clusters, n_boot=args.n_boot),
            "nll_gold": cluster_bootstrap_mean(nll, clusters, n_boot=args.n_boot),
            "agree": cluster_bootstrap_mean(raw[name]["agree"], clusters,
                                            n_boot=args.n_boot),
            "perplexity_gold": float(torch.exp(nll.mean())),
            "sequence_flip_rate": float(sum(flips) / len(flips)),
            "packed_bytes": packed["total"] if packed else None,
            "dense_bytes": dense_b,
            "compression_vs_loaded_dtype": (dense_b / packed["total"])
            if packed else None,
            "elapsed_s": round(time.time() - t1, 1),
        }
        a = out_arms[name]
        print("  %-16s KL %8.5f  gold %.4f  ppl %7.2f  seqflip %.4f  %.0fs"
              % (name, a["kl"]["point"], a["gold_top1"]["point"],
                 a["perplexity_gold"], a["sequence_flip_rate"], a["elapsed_s"]))

    delete_kl = out_arms["B-delete4"]["kl"]["point"]
    for name, a in out_arms.items():
        a["kl_recovery_vs_deletion"] = 1.0 - a["kl"]["point"] / delete_kl
        gd = out_arms["A-dense"]["gold_top1"]["point"]
        gx = out_arms["B-delete4"]["gold_top1"]["point"]
        a["gold_recovery_vs_deletion"] = ((a["gold_top1"]["point"] - gx) / (gd - gx)
                                          if abs(gd - gx) > 1e-9 else None)

    contrasts = []
    for a, b, why in [("C-int4-full", "B-delete4", "does the INT4 remnant recover?"),
                      ("C-int4-full", "A-dense", "residual gap to dense"),
                      ("G-shift1-int4", "C-int4-full", "adjacent wrong donor"),
                      ("G-shift6-int4", "C-int4-full", "distant wrong donor"),
                      ("G-rev-int4", "C-int4-full", "same weights, permuted placement"),
                      ("G-shift1-int4", "B-delete4", "is a wrong donor worse than nothing?"),
                      ("G-shift6-int4", "B-delete4", "is a wrong donor worse than nothing?"),
                      ("G-rev-int4", "B-delete4", "is a wrong donor worse than nothing?")]:
        row = {"arm": a, "vs": b, "question": why}
        for metric, key in (("kl", "kl"), ("gold_top1", "gold")):
            row[metric] = cluster_bootstrap_mean_pair(raw[a][key], raw[b][key],
                                                      clusters, n_boot=args.n_boot)
        contrasts.append(row)
        print("  %-16s vs %-14s dKL %+8.5f %s   dGold %+8.5f %s"
              % (a, b, row["kl"]["point"],
                 "SIG" if row["kl"]["excludes_zero"] else "ns ",
                 row["gold_top1"]["point"],
                 "SIG" if row["gold_top1"]["excludes_zero"] else "ns "))

    payload = {"audit": "sgqr_csgqr_dual", "block": "reference_arms",
               "split": args.split, "split_range": list(SPLITS[args.split]),
               "prompts": len(prompts), "reply_positions": len(clusters),
               "retired_layers": list(RETIRED), "n_groups_per_layer": N_GROUPS,
               "donor_maps": {k: (v if v is None else {str(a): b for a, b in v.items()})
                              for k, v in DM.items()},
               "n_boot": args.n_boot, "max_len": args.max_len,
               "unit_of_analysis": "reply position, bootstrap clustered by prompt",
               "arms": out_arms, "contrasts": contrasts,
               "environment": env_manifest()}
    (RES / ("reference_arms_%s.json" % args.split)).write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("wrote reference_arms_%s.json" % args.split)


if __name__ == "__main__":
    main()
