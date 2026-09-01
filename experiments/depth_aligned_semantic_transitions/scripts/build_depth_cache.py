"""Build a frozen-backbone pooled hidden-state cache. GPU forward passes only.

This is the missing reproducibility step for `FINDINGS_G3_SEEDS.md` section 2b:
that study consumed a cached tensor keyed ``SYM27`` that no committed script
produced. Everything downstream (`scripts/permutation_landscape.py`,
`scripts/trajectory_heterogeneity.py`) reads a cache written here.

No LoRA is loaded and no gradients are taken, so the cache describes the
UNTOUCHED model. That is the reference point section 2a needs: if a ridge probe on
this cache matches the trained projector, training read something already present.

Usage
-----
    python scripts/build_depth_cache.py --config configs/g3_s29_state_ordered.yaml \
        --num-checkpoints 4 --limit 2400 --out artifacts/depth_cache_k4_dev2400.pt

    # every usable layer, for the layer x stage profile
    python scripts/build_depth_cache.py --config configs/g3_s29_state_ordered.yaml \
        --all-layers --limit 2400 --out artifacts/depth_cache_all_dev2400.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.config import load_config  # noqa: E402
from dast.depth_cache import build_depth_cache  # noqa: E402
from dast.hf import load_frozen_base  # noqa: E402
from dast.model import infer_num_hidden_layers  # noqa: E402


def _free_accelerator_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
        used, total = (int(part.strip()) for part in out.stdout.strip().splitlines()[-1].split(","))
        return total - used
    except Exception:  # noqa: BLE001
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-checkpoints", type=int, default=None)
    parser.add_argument("--layers", default=None, help="explicit comma-separated 1-indexed layers")
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="cache every layer 1..num_hidden_layers-1 (the top is excluded: already post-final-norm)",
    )
    parser.add_argument(
        "--first-layer",
        type=int,
        default=1,
        help="lowest layer for --all-layers. Pass 0 to include the token-embedding "
        "output as the lexical control for the layer x stage profile.",
    )
    parser.add_argument("--readout", default="final_norm", choices=("raw", "final_norm"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-free-mib", type=int, default=3000)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.split.lower() == "test":
        raise SystemExit("refusing to build a cache from the sealed TEST split")

    if args.device.startswith("cuda"):
        free = _free_accelerator_mib()
        if 0 <= free < args.min_free_mib:
            raise SystemExit(
                f"only {free} MiB free on the accelerator, need {args.min_free_mib}; "
                "this GPU is shared with the desktop -- wait rather than fail into a squeeze"
            )
        print(f"accelerator headroom: {free} MiB free", flush=True)

    model, tokenizer = load_frozen_base(config)
    model.to(args.device)

    num_hidden_layers = infer_num_hidden_layers(model)
    layers = None
    if args.all_layers:
        # The top layer is excluded on purpose: hidden_states[num_hidden_layers] is
        # already post-final-norm, and DASTModel refuses it as a near-orthogonal
        # outlier. Including it would make one column incomparable to the rest.
        layers = list(range(args.first_layer, num_hidden_layers))
    elif args.layers:
        layers = [int(part) for part in args.layers.split(",")]

    print(
        f"model {config.model.name_or_path}: {num_hidden_layers} hidden layers; "
        f"readout={args.readout}; split={args.split}; limit={args.limit}",
        flush=True,
    )
    cache = build_depth_cache(
        model,
        tokenizer,
        config,
        layers=layers,
        num_checkpoints=args.num_checkpoints,
        split=args.split,
        limit=args.limit,
        readout=args.readout,
        batch_size=args.batch_size,
        device=args.device,
    )
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        revision = ""
    cache.metadata["git_revision"] = revision
    cache.metadata["built_by"] = "scripts/build_depth_cache.py"
    path = cache.save(args.out)
    print(
        f"\nwrote {path}\n  hidden {tuple(cache.hidden.shape)}\n  layers {cache.layers}\n"
        f"  metadata {json.dumps(cache.metadata, indent=2)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
