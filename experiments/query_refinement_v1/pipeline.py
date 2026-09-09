"""Bounded, source-bound qualification -> smoke -> open DEV, with no network IO."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F
from .core import ARMS, Spec
from .data import graph_rows, paired_summary
from .runtime import Runtime

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest():
    return {str(p.relative_to(HERE)): file_hash(p) for p in sorted(HERE.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json", ".txt")
            and "reports" not in p.parts and "__pycache__" not in p.parts}


def environment():
    import importlib.metadata
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "platform": platform.platform(), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "threads": torch.get_num_threads()}


def bind(model_dir, protocol):
    paths = sorted(p for p in model_dir.rglob("*") if p.is_file())
    weights = [p for p in paths if p.suffix in (".safetensors", ".bin")]
    if not weights:
        raise ValueError("No local checkpoint weights; do not substitute a toy model")
    for p in weights:
        with p.open("rb") as f:
            if f.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError("Unmaterialized LFS pointer, not checkpoint weights")
    # Hash every file: tokenizer, config, shard indexes and source weights included.
    return {"model_files": {str(p.relative_to(model_dir)): file_hash(p) for p in paths},
            "source_files": source_manifest(), "protocol": digest(protocol),
            "environment": environment()}


def require_receipt(receipt, binding, stage):
    if (receipt.get("stage") != stage or receipt.get("status") != "pass"
            or receipt.get("binding") != binding
            or receipt.get("sealed_evaluation_authorized") is not False):
        raise ValueError("Missing, failed, wrong-stage or stale qualification receipt")
    # Integrity, not a cryptographic attestation. Rerun qualification for untrusted receipts.
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != digest(body):
        raise ValueError("Receipt content hash mismatch")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def qualify(model, layer, rank):
    """No data bank access. Native forward is independent from our traversal."""
    model.eval().requires_grad_(False)
    g = torch.Generator().manual_seed(1729)
    ids = torch.randint(0, model.config.vocab_size, (1, 9), generator=g,
                        dtype=torch.long).to(next(model.parameters()).device)
    checks = {}
    with torch.no_grad():
        native = model(input_ids=ids, use_cache=False).logits
    for arm in ARMS:
        torch.manual_seed(17)
        rt = Runtime(model, Spec(arm, layer, rank)).freeze_backbone()
        with torch.no_grad():
            zero = rt(ids)
            torch.testing.assert_close(zero, native, atol=1e-5, rtol=1e-4)
            checks[f"{arm}/native_zero_max_error"] = float((zero-native).abs().max())
            if rt.update is not None:
                rt.update.up.weight.normal_(0, 0.08)
            full = rt(ids)
            if rt.update is not None and torch.equal(full, zero):
                raise AssertionError("Enabled intervention cannot reach logits")
            rt.reset()
            cached = torch.cat([rt(ids[:, a:b], use_cache=True)
                                for a, b in ((0, 2), (2, 5), (5, 9))], dim=1)
            torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-4)
            checks[f"{arm}/cache_max_error"] = float((cached-full).abs().max())
            changed = ids.clone()
            changed[:, 5:] = (changed[:, 5:]+1) % model.config.vocab_size
            torch.testing.assert_close(rt(changed)[:, :5], full[:, :5], atol=1e-5, rtol=1e-4)
            alt = Runtime(model, Spec(arm, layer, rank, "sdpa")).freeze_backbone()
            if rt.update is not None:
                alt.update.load_state_dict(rt.update.state_dict())
            torch.testing.assert_close(alt(ids), full, atol=1e-5, rtol=1e-4)
        if rt.update is not None:
            loss = rt(ids).square().mean()
            loss.backward()
            grads = [p.grad for p in rt.update.parameters()]
            if not all(x is not None and bool(torch.isfinite(x).all()) for x in grads):
                raise AssertionError("Nonfinite or missing adapter gradients")
            if not all(float(x.abs().sum()) > 0 for x in grads):
                raise AssertionError("Inactive trainable capacity after nonzero intervention")
            checks[f"{arm}/trainable_parameters"] = sum(p.numel() for p in rt.update.parameters())
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise AssertionError("Backbone was not frozen")
    return checks


def encode_rows(tokenizer, rows, max_tokens):
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS ID required")
    encoded = []
    for row in rows:
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(" " + row["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if not prompt or not answer or len(prompt) + len(answer) > max_tokens:
            raise ValueError("Token budget exceeded; no silent truncation or row dropping")
        encoded.append((row, prompt, answer))
    return encoded


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def train_arm(rt, rows, steps, seed, protocol, device):
    if rt.update is None:
        return {"optimizer_steps": 0, "answer_tokens": 0, "losses": [], "seconds": 0.0, "attention_macs": 0}
    optimizer = torch.optim.AdamW(rt.update.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=protocol["weight_decay"])
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    losses, tokens, macs = [], 0, 0
    start = clock(device)
    for step in range(steps):
        _, prompt, answer = rows[order[step % len(order)]]
        ids = torch.tensor([prompt + answer], device=device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        optimizer.zero_grad(set_to_none=True)
        logits = rt(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(rt.update.parameters(), protocol["gradient_clip"], error_if_nonfinite=True)
        if not math.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step()
        losses.append(float(loss.detach()))
        tokens += len(answer)
        c = rt.backbone.config
        reads = c.num_hidden_layers + int(rt.spec.arm in ("independent", "feedback"))
        macs += 2 * reads * c.num_attention_heads * ids.shape[1]**2 * c.head_dim
    if any(p.grad is not None or p.requires_grad for p in rt.backbone.parameters()):
        raise AssertionError("Unexpected trainable backbone")
    return {"optimizer_steps": steps, "answer_tokens": tokens, "losses": losses,
            "seconds": clock(device)-start, "attention_macs": macs}


def load_protocol():
    protocol = json.loads((HERE / "protocol.json").read_text())
    if (protocol["arms"] != list(ARMS) or protocol["backend"] != "eager"
            or protocol["dtype"] != "float32" or protocol["two_read_blend"] != 0.5
            or protocol["sealed_evaluation_authorized"] is not False):
        raise ValueError("Unsupported v1 protocol")
    for name, maximum in (("smoke_steps",25),("dev_steps",100),("max_tokens",512),
                          ("train_rows",64),("dev_rows",48),("max_new_tokens",16)):
        value = protocol[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"Unregistered {name} budget")
    seeds = protocol["training_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int for s in seeds):
        raise ValueError("Exactly three distinct training seeds required")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(protocol[key]) or protocol[key] <= 0:
            raise ValueError(f"Invalid {key}")
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("qualify", "smoke", "dev"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = load_protocol()
    import transformers
    if transformers.__version__ != protocol["transformers"]:
        raise ValueError("Use the pinned Transformers version, then requalify")
    if a.out.resolve().is_relative_to(a.model.resolve()):
        raise ValueError("Never write experiment results inside source checkpoint")
    if a.out.resolve().is_relative_to(HERE):
        raise ValueError("Use an external run directory to keep source fingerprints stable")
    binding = bind(a.model, protocol)
    binding["execution_device"] = a.device
    if a.stage != "qualify":
        if a.receipt is None:
            raise ValueError("Preceding successful receipt required")
        require_receipt(json.loads(a.receipt.read_text()), binding,
                        "qualify" if a.stage == "smoke" else "smoke")
    device = torch.device(a.device)
    if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] < 4096 * 1024**2:
        raise RuntimeError("R2 float32 preflight requires at least 4096 MiB free CUDA memory")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        attn_implementation="eager").to(device).eval()
    for key, value in protocol["target_config"].items():
        if getattr(model.config, key, None) != value:
            raise ValueError(f"Not the registered R2 checkpoint shape: {key}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from model")
    a.out.mkdir(parents=True, exist_ok=False)
    receipt = {"stage": a.stage, "status": "running", "binding": binding,
               "sealed_evaluation_authorized": False, "quality_claim_authorized": False}
    try:
        # Re-run mechanics even when a source-bound predecessor receipt exists.
        receipt["qualification"] = qualify(model, protocol["layer"], protocol["rank"])
        if a.stage != "qualify":
            train = graph_rows("train", protocol["train_rows"], protocol["data_seed"])
            dev = graph_rows("dev", protocol["dev_rows"], protocol["data_seed"])
            encoded = encode_rows(tokenizer, train, protocol["max_tokens"])
            encoded_dev = encode_rows(tokenizer, dev, protocol["max_tokens"])
            write_json(a.out / "open_data.json", {"train": train, "dev": dev})
            receipt["data_sha256"] = digest({"train": train, "dev": dev})
            results, records = [], []
            seeds = protocol["training_seeds"] if a.stage == "dev" else protocol["training_seeds"][:1]
            for seed in seeds:
                for arm in ARMS:
                    torch.manual_seed(seed)
                    rt = Runtime(model, Spec(arm, protocol["layer"], protocol["rank"])).freeze_backbone()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    steps = protocol["dev_steps"] if a.stage == "dev" else protocol["smoke_steps"]
                    metrics = train_arm(rt, encoded, steps, seed, protocol, device)
                    if a.stage == "dev":
                        start = clock(device)
                        for row, prompt, _ in encoded_dev:
                            output = rt.generate(torch.tensor([prompt], device=device),
                                                 max_new_tokens=protocol["max_new_tokens"], eos=(tokenizer.eos_token_id,))
                            answer = tokenizer.decode(output, skip_special_tokens=True).strip()
                            records.append({"arm": arm, "seed": seed, "id": row["id"], "hops": row["hops"],
                                            "prediction": answer, "expected": row["answer"],
                                            "correct": answer == row["answer"], "generated_ids": output})
                        metrics["dev_seconds"] = clock(device)-start
                    metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    if rt.update is not None:
                        torch.save(rt.update.state_dict(), a.out / f"{arm}-{seed}-adapter.pt")
                    results.append({"arm": arm, "seed": seed, **metrics})
                    del rt
            write_json(a.out / "training.json", results)
            if records:
                write_json(a.out / "predictions.json", records)
                write_json(a.out / "summary.json", paired_summary(records))
                write_json(a.out / "by_hop.json", {h: paired_summary([r for r in records if r["hops"] == h]) for h in range(1, 5)})
        receipt["status"] = "pass"
    except Exception as error:
        receipt["status"], receipt["error"] = "fail", f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["outputs"] = {p.name: file_hash(p) for p in sorted(a.out.iterdir()) if p.is_file()}
        receipt["receipt_sha256"] = digest(receipt)
        write_json(a.out / "receipt.json", receipt)
        model.zero_grad(set_to_none=True)
    print(json.dumps({"stage": a.stage, "status": receipt["status"], "out": str(a.out)}))


if __name__ == "__main__":
    main()
