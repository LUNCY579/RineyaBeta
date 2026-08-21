#!/usr/bin/env python3
"""Export a training checkpoint's LoRA weights as a PEFT-loadable directory.

`heal.py` checkpoints store `adapter_state` (the output of
`get_peft_model_state_dict`) inside a `.pt` alongside optimizer state, RNG streams
and provenance -- everything needed to *resume*. But `run_gates.py --adapter` and
`PeftModel.from_pretrained` want a directory with `adapter_config.json` and
`adapter_model.safetensors`. Without this bridge a finished run produces an adapter
that cannot be evaluated, which is a silly place to discover a gap.

The LoRA hyperparameters are read back out of the checkpoint's stored config rather
than passed in again, so an exported adapter cannot silently disagree with the run
that produced it.

Usage:
  python scripts/train/export_adapter.py artifacts/heal_runs/b0 --out artifacts/adapters/b0
  python scripts/train/export_adapter.py <path/to/step_000001003.pt> --out ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path) -> str:
    """Repo-relative if possible, absolute otherwise. Never raises."""
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve_checkpoint(target: Path) -> Path:
    if target.is_file():
        return target
    ckpt_dir = target / "checkpoints" if (target / "checkpoints").is_dir() else target
    pointer = ckpt_dir / "latest.json"
    if pointer.exists():
        name = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    payloads = sorted(ckpt_dir.glob("step_*.pt"))
    if not payloads:
        raise SystemExit(f"no checkpoint found under {target}")
    return payloads[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="run dir, checkpoints dir, or a .pt file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None,
                    help="override the base model path recorded in the checkpoint")
    args = ap.parse_args()

    path = resolve_checkpoint(Path(args.checkpoint).resolve())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    lora = cfg["lora"]
    base = args.base or cfg["model"]["path"]
    position = payload["position"]

    from peft import LoraConfig
    peft_config = LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.0)), bias="none",
        task_type="CAUSAL_LM", target_modules=list(lora["target_modules"]),
        base_model_name_or_path=str(base))

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    peft_config.save_pretrained(str(out))

    state = payload["adapter_state"]
    # PeftModel.from_pretrained expects the `base_model.model.` prefix that
    # get_peft_model_state_dict strips, and `.weight` suffixes intact.
    from safetensors.torch import save_file
    tensors = {k: (v.contiguous() if torch.is_tensor(v) else v)
               for k, v in state.items() if torch.is_tensor(v)}
    save_file(tensors, str(out / "adapter_model.safetensors"),
              metadata={"format": "pt"})

    (out / "export_manifest.json").write_text(json.dumps({
        "source_checkpoint": rel(path),
        "global_step": position["global_step"],
        "epoch": position["epoch"],
        "rows_seen": position["rows_seen"],
        "supervised_tokens_seen": position["supervised_tokens_seen"],
        "base_model": str(base),
        "lora": {k: lora[k] for k in ("r", "alpha", "dropout", "target_modules")
                 if k in lora},
        "mix_sha256": payload.get("mix_sha256"),
        "plan_hash": payload.get("plan_hash"),
        "metrics_at_save": payload.get("metrics"),
        "tensors": len(tensors),
        "trainable_params": int(sum(v.numel() for v in tensors.values())),
    }, indent=2), encoding="utf-8")

    print(f"source     : {path.name}  (step {position['global_step']:,d})")
    print(f"base model : {base}")
    print(f"lora       : r={lora['r']} alpha={lora['alpha']} "
          f"targets={len(lora['target_modules'])}")
    print(f"tensors    : {len(tensors)}  "
          f"params {sum(v.numel() for v in tensors.values()):,d}")
    print(f"rows seen  : {position['rows_seen']:,d}")
    print(f"wrote {rel(out)}")


if __name__ == "__main__":
    main()
