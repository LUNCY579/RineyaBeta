"""Measure cosine(∇L_LM, ∇L_state) on an existing V1 run. No optimizer step.

    python scripts/gradient_conflict.py --run runs/g3_s29_state_ordered --batches 2

Refuses to start if a DAST trainer is already using the GPU.
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
from dast.grad_conflict import conflict_from_losses  # noqa: E402
from dast.losses import compute_loss  # noqa: E402
from dast.model import DASTModel  # noqa: E402
from dast.semantic import FrozenTextTargetEncoder  # noqa: E402
from dast.training import _encode_targets, _move_batch, build_dataloaders  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def trainer_running() -> bool:
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        for proc in psutil.process_iter(attrs=["cmdline"]):
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "dast.cli" in cmd and "train" in cmd:
                return True
            if "resume_runs.py" in cmd:
                return True
        return False
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        blob = out.stdout.lower()
        return ("dast.cli" in blob and "train" in blob) or ("resume_runs.py" in blob)
    except Exception:  # noqa: BLE001
        return False


def load_trained(run_dir: Path, config):
    from peft import PeftModel

    from dast.hf import build_hf_components

    enabled = config.lora.enabled
    config.lora.enabled = False
    base, tokenizer, _info = build_hf_components(config, for_training=True)
    config.lora.enabled = enabled
    adapter = run_dir / "adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"no adapter at {adapter}")
    base = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model = DASTModel(
        base,
        checkpoint_fractions=config.model.checkpoint_fractions,
        state_dim=config.semantic.state_dim,
        projector_hidden_dim=config.semantic.projector_hidden_dim,
        pooling=config.model.pooling,
        checkpoint_readout=config.model.checkpoint_readout,
    )
    auxiliary = torch.load(run_dir / "dast_auxiliary.pt", map_location="cpu", weights_only=False)
    model.state_projector.load_state_dict(auxiliary["state_projector"])
    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 LM/state gradient cosine (no training)")
    parser.add_argument("--run", required=True, help="completed V1 run directory")
    parser.add_argument("--config", help="YAML used for that arm; default: run_dir sibling inference")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    if trainer_running():
        print("REFUSE: a DAST trainer is running; B2 deferred (GPU exclusivity).")
        return 2

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config_path = Path(args.config) if args.config else None
    if config_path is None:
        arm = manifest["arm"]
        seed = int(manifest["seed"])
        guess = ROOT / "configs" / f"g3_s{seed}_{arm}.yaml"
        if not guess.exists() and seed == 17:
            guess = ROOT / "configs" / f"g2_budget_{arm}.yaml"
        if not guess.exists() and arm == "state_final_repeated" and seed == 17:
            guess = ROOT / "configs" / "g3_s17_state_final_repeated.yaml"
        config_path = guess
    config = load_config(config_path)
    model, tokenizer = load_trained(run_dir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    embedding = model.base_model.get_input_embeddings()
    encoder = FrozenTextTargetEncoder(
        embedding,
        tokenizer,
        state_dim=config.semantic.state_dim,
        projection_seed=config.semantic.target_projection_seed,
        max_tokens=config.semantic.target_max_tokens,
    )
    train_loader, _eval_loader, _paths = build_dataloaders(config, tokenizer)
    rows = []
    for index, raw in enumerate(train_loader):
        if index >= args.batches:
            break
        batch = _move_batch(raw, device)
        targets = _encode_targets(
            encoder,
            batch["training_target_texts"],
            device=device,
            centering=config.semantic.target_centering,
            reference_texts=batch["canonical_target_texts"],
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            prompt_mask=batch["prompt_mask"],
        )
        breakdown = compute_loss(
            output,
            config=config,
            training_targets=targets,
            training_group_ids=batch.get("training_group_ids"),
            operator_bank=None,
        )
        report = conflict_from_losses(
            breakdown.lm, breakdown.state, model.named_parameters()
        )
        report["batch_index"] = index
        rows.append(report)
        print(
            f"batch {index}: cosine={report['global_cosine']:.4f} "
            f"ratio={report['grad_norm_ratio_state_over_lm']:.4f} "
            f"frac_neg={report['fraction_modules_negative_cosine']:.3f}"
        )
    mean_cosine = sum(r["global_cosine"] for r in rows) / len(rows) if rows else float("nan")
    payload = {
        "run": str(run_dir),
        "arm": manifest.get("arm"),
        "seed": manifest.get("seed"),
        "config_sha256": manifest.get("config_sha256"),
        "n_batches": len(rows),
        "mean_global_cosine": mean_cosine,
        "batches": rows,
        "claim_boundary": "B2 diagnosis only. Not a capability result. TEST sealed.",
    }
    out = Path(args.out) if args.out else ROOT / "artifacts" / f"b2_{run_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
