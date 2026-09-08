"""Local-only artifact identity and fail-closed receipt validation."""
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_COMMIT = "3532c7bc8dbdd48ef8801ba6080cb80b0e90379a"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def code_identity():
    files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    files += [ROOT / "protocol.json"]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in files})


def environment(device="cpu"):
    import torch
    versions = {}
    for name in ("torch", "transformers", "safetensors", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "device": device, "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()}


def checkpoint_inventory(path):
    """Only checkpoint assets are read. Never crawl corpora or evaluation folders."""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Checkpoint must be an existing local directory")
    config = root / "config.json"
    if not config.is_file() or config.is_symlink():
        raise FileNotFoundError("Local checkpoint config.json is missing")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if cfg.get("model_type") != "qwen3" or cfg.get("auto_map"):
        raise ValueError("Only local native Qwen3 checkpoints, without remote code")
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.is_file():
        weights = [single]
    elif index.is_file() and not index.is_symlink():
        idx = json.loads(index.read_text(encoding="utf-8"))
        names = set(idx.get("weight_map", {}).values())
        if not names or any(Path(n).name != n or not n.endswith(".safetensors") for n in names):
            raise ValueError("Invalid safetensors shard index")
        weights = [root / name for name in sorted(names)]
    else:
        raise FileNotFoundError("Real weights missing: config/tokenizer alone cannot qualify Rineya")
    files = weights + [root / name for name in (
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
    ) if (root / name).is_file()]
    hashes = {}
    for p in files:
        if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:
            raise ValueError(f"Missing, empty or symlinked checkpoint asset: {p.name}")
        if p in weights:
            with p.open("rb") as f:
                if f.read(80).startswith(b"version https://git-lfs.github.com/spec"):
                    raise ValueError("Unresolved Git LFS weight pointer")
        hashes[p.name] = {"sha256": file_hash(p), "bytes": p.stat().st_size}
    return {"id": digest(hashes), "files": hashes, "config": cfg}


def new_run(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be a simple, new identifier (no path components)")
    parent = ROOT / "runs"
    if parent.is_symlink():
        raise ValueError("Run directory cannot be a symlink")
    parent.mkdir(exist_ok=True)
    out = parent / run_id
    out.mkdir(exist_ok=False)
    return out


def write_json(path, value):
    # Exclusive creation preserves negative/failed receipts; never overwrite a run.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def require_receipt(receipt, inventory, env, backend, block_size):
    expected = {
        "stage": "real_model_qualification", "passed": True,
        "checkpoint_id": inventory["id"], "code_identity": code_identity(),
        "environment": env, "backend": backend, "block_size": block_size,
        "licenses_training": False, "licenses_sealed_evaluation": False,
    }
    for name, value in expected.items():
        if name not in receipt or receipt[name] != value:
            raise ValueError(f"Receipt mismatch/missing field: {name}")
    required = ("native_global", "global_stream", "gate_identity", "local_stream", "sink_stream")
    checks = receipt.get("checks", {})
    if not all(checks.get(k, {}).get("passed") is True for k in required):
        raise ValueError("Incomplete real-model qualification checks")
    # Integrity binding detects accidental edits; not an authorization signature.
    payload = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != digest(payload):
        raise ValueError("Receipt integrity mismatch")
