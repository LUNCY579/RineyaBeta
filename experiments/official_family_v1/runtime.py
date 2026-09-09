"""Read-only official-core identity, explicit Qwen3 loading and evidence binding.

Only the explicit restore CLI performs network I/O. Inventory is not model qualification.
The registered releases use their original unsharded model.safetensors files.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSFORMERS_VERSION = "4.57.1"
SHAPE = {
    "model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 1024,
    "intermediate_size": 3072, "num_attention_heads": 16,
    "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
    "max_position_embeddings": 40960, "tie_word_embeddings": True,
    "bos_token_id": 151643, "eos_token_id": 151645,
}
UNIQUE_PARAMETERS = 596049920
REQUIRED = ("CORE.json", "config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json")
OPTIONAL = ("generation_config.json", "chat_template.jinja", "special_tokens_map.json",
            "added_tokens.json", "vocab.json", "merges.txt")


class FamilyError(ValueError):
    """Fail closed: no checkpoint, runtime or evidence substitution."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FamilyError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise FamilyError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=invalid)


def registry():
    return read_json(HERE / "profiles.json")


def get_profile(name):
    data = registry()
    if name not in data["profiles"]:
        raise FamilyError("Choose an exact registered dense-core profile: " +
                          ", ".join(data["profiles"]))
    return {"name": name, **data["profiles"][name]}


def hash_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FamilyError(f"Expected materialized regular file: {path}")
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    if path.is_symlink() or stamp(before) != stamp(path.stat()):
        raise FamilyError(f"File changed while hashing: {path}")
    return {"sha256": h.hexdigest(), "bytes": before.st_size}


def normalize_config(raw):
    """Translate only the documented default-RoPE serialization, in memory.

    Halo's newer exporter writes rope_parameters; never let 4.57.1 silently
    ignore this key and choose a default theta. Scaled/custom RoPE is rejected.
    """
    cfg = dict(raw)
    for key, expected in SHAPE.items():
        actual = cfg.get(key)
        if actual != expected or type(actual) is not type(expected):
            raise FamilyError(f"Unregistered dense-core configuration: {key}")
    if cfg.get("architectures") != ["Qwen3ForCausalLM"]:
        raise FamilyError("A stock Qwen3ForCausalLM architecture is required")
    for key in ("auto_map", "quantization_config", "abstain_logit_bias", "peft_config"):
        if key in cfg:
            raise FamilyError(f"Custom/quantized/policy configuration is not supported: {key}")
    if (cfg.get("attention_bias", False) is not False
            or cfg.get("attention_dropout", 0) != 0
            or cfg.get("use_sliding_window", False) is not False
            or cfg.get("sliding_window") is not None
            or cfg.get("rope_scaling") is not None
            or cfg.get("hidden_act") != "silu"
            or cfg.get("rms_norm_eps") != 1e-6):
        raise FamilyError("Requires original zero-dropout full attention and unscaled RoPE")
    types = cfg.get("layer_types")
    if types is not None and types != ["full_attention"] * SHAPE["num_hidden_layers"]:
        raise FamilyError("Unexpected layer types")
    modern = cfg.pop("rope_parameters", None)
    if modern is not None:
        if modern != {"rope_type": "default", "rope_theta": 1000000}:
            raise FamilyError("Only the registered default rope_parameters are supported")
        if "rope_theta" in cfg and cfg["rope_theta"] != modern["rope_theta"]:
            raise FamilyError("Conflicting RoPE representations")
        cfg["rope_theta"] = modern["rope_theta"]
    if cfg.get("rope_theta") != 1000000:
        raise FamilyError("Missing or altered original RoPE theta")
    cfg["rope_scaling"] = None
    return cfg


def inventory(root, profile_name):
    profile = get_profile(profile_name)
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise FamilyError("Supply a materialized core directory, not a model ID or symlink")
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FamilyError("Incomplete official checkpoint; missing " + ", ".join(missing) +
                          ". Restore the pinned private HF core; metadata/GGUF/adapters are not replacements.")
    for name in ("adapter_config.json", "SYSTEM.json", "model.safetensors.index.json"):
        if (root / name).exists():
            raise FamilyError(f"Unregistered adapter/system/sharded export: {name}")
    names = [*REQUIRED, *(name for name in OPTIONAL if (root / name).exists())]
    # Stat the whole runtime asset set before and after hashing, not each file only.
    def stamps():
        return {name: (hash_path.stat().st_size, hash_path.stat().st_mtime_ns,
                       hash_path.stat().st_ctime_ns, hash_path.stat().st_ino)
                for name in names for hash_path in [root / name]}
    before = stamps()
    for name in names:
        with (root / name).open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
                raise FamilyError(f"Unresolved Git LFS pointer: {name}")
    raw = read_json(root / "config.json")
    normalized = normalize_config(raw)
    core = read_json(root / "CORE.json")
    if core.get("core_id") != profile["core_id"]:
        raise FamilyError("CORE.json does not identify the selected official core")
    checks = core.get("checks", {})
    if (checks.get("weight_sha256") != profile["weight_sha256"]
            or checks.get("weight_bytes") != profile["weight_bytes"]):
        raise FamilyError("CORE weight provenance differs from the pinned registry")
    tc = read_json(root / "tokenizer_config.json")
    if tc.get("auto_map") or tc.get("tokenizer_class") not in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
        raise FamilyError("Only the built-in Qwen2 tokenizer implementation is supported")
    files = {name: hash_file(root / name) for name in sorted(names)}
    expected_weight = {"sha256": profile["weight_sha256"], "bytes": profile["weight_bytes"]}
    if files["model.safetensors"] != expected_weight:
        raise FamilyError("Weights do not match the official release (no random/stock-Qwen/R2 fallback)")
    for name, expected in profile.get("reference_sha256", {}).items():
        if files[name]["sha256"] != expected:
            raise FamilyError(f"Official reference hash mismatch: {name}")
    for name, expected in profile.get("reference_git_blobs", {}).items():
        data = (root / name).read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_sha != expected:
            raise FamilyError(f"Official Git blob mismatch: {name}; do not rewrite checkpoint metadata")
    from safetensors import safe_open
    try:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as tensors:
            if not list(tensors.keys()):
                raise FamilyError("Empty safetensors checkpoint")
    except Exception as error:
        raise FamilyError("Invalid safetensors checkpoint") from error
    if before != stamps() or names != [*REQUIRED, *(n for n in OPTIONAL if (root / n).exists())]:
        raise FamilyError("Checkpoint changed during inventory")
    return {"schema": "rineya.family-assets/1", "profile": profile,
            "registry_sha256": digest(registry()), "files": files,
            "identity": digest(files), "normalized_config": normalized,
            "config_translation": "default_rope_parameters_to_legacy" if "rope_parameters" in raw else "none",
            "official_weight_hash_verified": True,
            "tokenizer_reference_hash_verified": "tokenizer.json" in profile.get("reference_sha256", {}),
            "evidence_class": "ASSET_VERIFICATION_ONLY", "licenses_training": False}


def validate_tokenizer(tokenizer, config):
    # Qwen vocab matrices can contain padding rows. len(tokenizer)==vocab_size
    # is NOT a valid universal invariant; all actual IDs must fit instead.
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    if not ids or any(type(i) is not int or not 0 <= i < config.vocab_size for i in ids):
        raise FamilyError("Tokenizer IDs do not fit the model's embedding rows")
    if len(ids) != len(set(ids)):
        raise FamilyError("Duplicate tokenizer vocabulary IDs")
    if tokenizer.eos_token_id != config.eos_token_id or vocab.get("<|im_end|>") != config.eos_token_id:
        raise FamilyError("EOS mapping differs from the registered family checkpoint")
    if tokenizer.pad_token_id != 151643 or vocab.get("<|endoftext|>") != 151643:
        raise FamilyError("Unexpected tokenizer padding/end-of-text mapping")
    return {"entries": len(vocab), "maximum_id": max(ids), "embedding_rows": config.vocab_size,
            "vocabulary_sha256": digest(vocab), "eos_token_ids": [config.eos_token_id]}


def runtime_environment(device):
    if device not in ("cpu", "cuda:0"):
        raise FamilyError("Choose cpu or cuda:0 explicitly; automatic dispatch is forbidden")
    import torch
    if device == "cuda:0" and not torch.cuda.is_available():
        raise FamilyError("CUDA unavailable; no automatic CPU fallback")
    versions = {name: importlib.metadata.version(name)
                for name in ("transformers", "safetensors", "tokenizers")}
    if versions["transformers"] != TRANSFORMERS_VERSION:
        raise FamilyError(f"Use transformers=={TRANSFORMERS_VERSION}; other versions need requalification")
    return {"python": platform.python_version(), "torch": str(torch.__version__), **versions,
            "platform": platform.platform(), "device": device,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda:0" else platform.processor(),
            "threads": torch.get_num_threads(), "dtype": "float32", "backend": "eager",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def setup_runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def check_loading_info(info):
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs", "error_messages")):
        raise FamilyError(f"Incomplete/non-exact pretrained state load: {info}")


def load_model(root, profile_name, device, expected_assets):
    """Load exact pretrained state; an inventory is never accepted as qualification."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen2TokenizerFast
    runtime_environment(device)
    observed = inventory(root, profile_name)
    if observed != expected_assets:
        raise FamilyError("Checkpoint drift before loading")
    if device == "cuda:0" and torch.cuda.mem_get_info(0)[0] < 6144 * 2**20:
        raise FamilyError("Need 6144 MiB free CUDA memory; do not interrupt other workloads")
    config = Qwen3Config.from_dict(observed["normalized_config"])
    tokenizer = Qwen2TokenizerFast.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    token_info = validate_tokenizer(tokenizer, config)
    model, info = Qwen3ForCausalLM.from_pretrained(
        root, config=config, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation="eager",
        output_loading_info=True)
    check_loading_info(info)
    if type(model) is not Qwen3ForCausalLM or getattr(model, "is_quantized", False):
        raise FamilyError("Unexpected wrapper/quantized model")
    if sum(p.numel() for p in model.parameters()) != UNIQUE_PARAMETERS:
        raise FamilyError("Unexpected unique parameter count")
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise FamilyError("Original embedding/head tying was not preserved")
    if (model.config._attn_implementation != "eager"
            or model.config.rope_theta != 1000000
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise FamilyError("Runtime configuration/dtype drift")
    model.to(device).eval().requires_grad_(False)
    return model, tokenizer, token_info


def source_manifest(experiment_dir):
    result = {}
    for root in (HERE, Path(experiment_dir).resolve()):
        for path in sorted(root.rglob("*")):
            if (path.is_file() and path.suffix in (".py", ".json", ".txt")
                    and not any(p in ("__pycache__", "reports", "evidence") for p in path.relative_to(root).parts)):
                result[root.name + "/" + path.relative_to(root).as_posix()] = hash_file(path)
    return result


def binding(root, profile_name, experiment_dir, protocol, device):
    return {"schema": "rineya.family-binding/1", "experiment": protocol["experiment"],
            "assets": inventory(root, profile_name), "source_files": source_manifest(experiment_dir),
            "protocol": protocol, "runtime": runtime_environment(device)}


def require_receipt(receipt, bound, stage, receipt_path=None):
    if (receipt.get("schema") != "rineya.family-receipt/1"
            or receipt.get("stage") != stage or receipt.get("status") != "pass"
            or receipt.get("binding") != bound
            or receipt.get("evidence_class") != "PRETRAINED_OFFICIAL_FAMILY_EXPERIMENT"
            or receipt.get("sealed_evaluation_authorized") is not False
            or receipt.get("quality_claim_authorized") is not False):
        raise FamilyError("Missing, stale or wrong-model/stage receipt; R2 evidence is not transferable")
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != digest(body):
        raise FamilyError("Receipt hash mismatch")
    experiment = bound["experiment"]
    checks = receipt.get("qualification")
    def finite_nonnegative(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    if experiment == "query_refinement_official_family_v1":
        if stage not in ("qualify", "smoke") or not isinstance(checks, dict):
            raise FamilyError("Invalid query predecessor stage or missing mechanical checks")
        expected_next = {"qualify": "smoke", "smoke": "dev"}[stage]
        for arm in ("baseline", "adapter", "independent", "feedback"):
            for metric in ("native_zero_max_error", "cache_max_error"):
                if not finite_nonnegative(checks.get(f"{arm}/{metric}")):
                    raise FamilyError("Incomplete native/cache qualification")
            if arm != "baseline" and checks.get(f"{arm}/trainable_parameters") != 2064:
                raise FamilyError("Adapter capacity differs from the registered 2064 parameters")
        if stage == "smoke":
            seed = bound["protocol"]["training_seeds"][0]
            budgets = receipt.get("training_budget")
            if not isinstance(budgets, list) or len(budgets) != 4:
                raise FamilyError("Missing bounded optimization smoke evidence")
            for row, arm in zip(budgets, ("baseline", "adapter", "independent", "feedback")):
                expected_steps = 0 if arm == "baseline" else bound["protocol"]["smoke_steps"]
                if row.get("arm") != arm or row.get("seed") != seed or row.get("optimizer_steps") != expected_steps:
                    raise FamilyError("Smoke run did not cover the registered arms and budgets")
            if not all(type(row.get("answer_tokens")) is int and row["answer_tokens"] > 0 for row in budgets[1:]):
                raise FamilyError("Smoke did not process answer tokens")
            if len({row["answer_tokens"] for row in budgets[1:]}) != 1:
                raise FamilyError("Smoke answer-token budgets differ between trainable arms")
            if "training.json" not in receipt.get("outputs", {}):
                raise FamilyError("Missing smoke training artifact")
    elif experiment == "composition_qrr_official_family_v1":
        expected_next = "diagnose"
        if (stage != "qualify" or not isinstance(checks, list) or len(checks) != 3
                or receipt.get("licenses_training") is not False
                or receipt.get("qrr_native_qualified") is not False):
            raise FamilyError("Invalid frozen-only diagnostic qualification")
        for probe in checks:
            steps = probe.get("steps", [])
            if len(steps) != 4:
                raise FamilyError("Incomplete frozen cache trace")
            for i, row in enumerate(steps):
                if (row.get("step") != i or row.get("parity_argmax_equal") is not True
                        or not finite_nonnegative(row.get("parity_max_abs"))):
                    raise FamilyError("Incomplete frozen cache parity evidence")
    else:
        raise FamilyError("Unknown experiment evidence scope")
    if receipt.get("licenses_next_stage") != expected_next:
        raise FamilyError("Receipt does not license the requested bounded transition")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise FamilyError("Missing output manifest")
    for name in outputs:
        if Path(name).name != name or name in (".", "..", "receipt.json") or "\\" in name:
            raise FamilyError("Unsafe receipt output path")
    if receipt_path is not None:
        parent = Path(receipt_path).resolve().parent
        for name, expected in outputs.items():
            if hash_file(parent / name) != expected:
                raise FamilyError("Predecessor output artifact changed or is missing")


def seal_receipt(receipt):
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    return {**body, "receipt_sha256": digest(body)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def check_output(output, model_root, experiment_dir):
    out = Path(output).resolve()
    for source in (Path(model_root).resolve(), HERE, Path(experiment_dir).resolve()):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise FamilyError("Output must be separate from checkpoint and experiment source")
