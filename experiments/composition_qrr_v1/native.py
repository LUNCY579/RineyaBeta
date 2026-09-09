"""Separately qualified, local-only frozen R2 diagnostic path.

No automatic remote download, remote modelling code, optimizer, sealed reader,
QRR installation or Supa-native-lock change exists in this module.
"""
from __future__ import annotations
from pathlib import Path
import platform
import time
from .identity import (ContractError, digest, inventory, positive_int, read_json,
                       safe_run_dir, source_identity, write_new)
from .instrument import ARMS, audit_bank, prompt_for, score

R2_SHAPE = {"num_hidden_layers": 28, "hidden_size": 1024, "intermediate_size": 3072,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 32768}
PROBES = ("The glass is amber. Its color is", "Row 1: A. Row 2: B. After A is",
          "This is a mechanical parity probe, not a quality benchmark.")


def validate_contract(contract):
    if contract.get("schema") != "qrr.native-contract/1" or contract.get("status") != "REVIEWED_FROZEN_ONLY":
        raise ContractError("A separately reviewed frozen-only native contract is required")
    if contract.get("target") != "R2" or contract.get("prompt_protocol") != "raw_completion":
        raise ContractError("This revision supports R2 raw completion only")
    if not isinstance(contract.get("reviewed_by"), str) or not contract["reviewed_by"].strip():
        raise ContractError("Missing review provenance")
    if contract.get("source_identity") != source_identity():
        raise ContractError("Contract does not bind this source revision")
    assets = contract.get("assets")
    if (not isinstance(assets, dict) or not isinstance(assets.get("files"), dict)
            or not assets["files"] or assets.get("identity") != digest(assets["files"])):
        raise ContractError("Missing/inconsistent independently reviewed asset fingerprint")
    runtime = contract["runtime"]
    if runtime.get("device") != "cuda:0" or runtime.get("dtype") not in ("float32", "float16", "bfloat16"):
        raise ContractError("Explicit CUDA:0 precision required; no silent CPU fallback")
    if runtime.get("class") != "Qwen3ForCausalLM" or runtime.get("backend") != "eager":
        raise ContractError("Only standard Qwen3 eager is qualified by this path")
    for key in ("torch", "transformers", "python", "cuda_runtime", "cuda_device_name"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ContractError(f"Exact {key} version is required")
    limits = contract["limits"]
    for name, maximum in (("prompt_tokens", 4096), ("new_tokens", 24), ("cases", 768)):
        positive_int(limits[name], name, maximum)
    if contract.get("eos_token_ids") != [32744]:
        raise ContractError("R2 raw-completion EOS must be explicitly registered as [32744]")
    return contract


def open_model(root, contract):
    validate_contract(contract)
    observed = inventory(root)
    if observed["files"] != contract["assets"]["files"]:
        raise ContractError("Checkpoint differs from reviewed inventory")
    import torch
    import transformers
    runtime = contract["runtime"]
    if (torch.__version__ != runtime["torch"] or transformers.__version__ != runtime["transformers"]
            or platform.python_version() != runtime["python"]):
        raise ContractError("Runtime drift from reviewed contract")
    if not torch.cuda.is_available():
        raise ContractError("CUDA unavailable; no fallback")
    if torch.version.cuda != runtime["cuda_runtime"] or torch.cuda.get_device_name(0) != runtime["cuda_device_name"]:
        raise ContractError("GPU/runtime differs from reviewed contract")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    free, total = torch.cuda.mem_get_info(0)
    if free < 6144 * 2**20:
        raise ContractError("Need 6144 MiB free; do not interrupt other workloads")
    torch.cuda.set_per_process_memory_fraction(min(6144 * 2**20 / total, .9), 0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=getattr(torch, runtime["dtype"]), attn_implementation="eager").to("cuda:0")
    model.eval().requires_grad_(False)
    if type(model).__name__ != runtime["class"]:
        raise ContractError("Unexpected runtime model class")
    for key, value in R2_SHAPE.items():
        if getattr(model.config, key, None) != value:
            raise ContractError(f"R2 config mismatch: {key}")
    if getattr(model.config, "abstain_logit_bias", 0) or getattr(model.config, "auto_map", None):
        raise ContractError("Custom wrapper/policy is outside this contract")
    if model.config.eos_token_id != 32744 or len(tokenizer) != 32768:
        raise ContractError("Tokenizer/EOS mismatch")
    if sum(p.numel() for p in model.parameters()) != 474021888:
        raise ContractError("Actual unique parameter count differs from R2")
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ContractError("R2 embedding/head tying was not preserved")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ContractError("Requested attention backend was not installed")
    if contract["limits"]["prompt_tokens"] + contract["limits"]["new_tokens"] > model.config.max_position_embeddings:
        raise ContractError("Contract exceeds native position limit")
    return model, tokenizer, observed


def encode(tokenizer, text, limit):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > limit:
        raise ContractError("Empty or over-budget prompt; truncation is forbidden")
    return ids


def greedy_trace(model, tokenizer, ids, new_tokens, eos_ids, check_parity=False):
    import torch
    positive_int(new_tokens, "new_tokens", 24)
    if not ids:
        raise ContractError("Empty native input")
    device = next(model.parameters()).device
    prefix = torch.tensor([ids], dtype=torch.long, device=device)
    input_ids, past, generated, records = prefix, None, [], []
    unknown = tokenizer.encode("UNKNOWN", add_special_tokens=False)
    unknown = unknown[0] if len(unknown) == 1 else None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(new_tokens):
            out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise ContractError("Non-finite native logits")
            if check_parity:
                full = model(input_ids=prefix, use_cache=False).logits[0, -1].float()
                atol = 2e-5 if next(model.parameters()).dtype == torch.float32 else .125
                if not torch.allclose(logits, full, atol=atol, rtol=1e-3) or logits.argmax() != full.argmax():
                    raise ContractError("Full/cached parity failed")
            token = int(logits.argmax())
            row = {"step": step, "chosen_id": token, "chosen_logit": float(logits[token])}
            if check_parity:
                row.update(parity_argmax_equal=True, parity_max_abs=float((logits - full).abs().max()))
            if unknown is not None:
                if not 0 <= unknown < logits.numel():
                    raise ContractError("UNKNOWN token outside model vocabulary")
                other = logits.clone()
                other[unknown] = -torch.inf
                best = int(other.argmax())
                row.update(unknown_id=unknown, unknown_logit=float(logits[unknown]),
                           best_other_id=best, best_other_logit=float(other[best]))
            records.append(row)
            generated.append(token)
            if token in eos_ids:
                break
            past = out.past_key_values
            if past is None:
                raise ContractError("Requested cache was not returned")
            input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
            prefix = torch.cat((prefix, input_ids), dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"text": tokenizer.decode(generated, skip_special_tokens=True), "token_ids": generated,
            "input_tokens": len(ids), "output_tokens_including_eos": len(generated),
            "termination": "eos" if generated[-1] in eos_ids else "budget",
            "elapsed_seconds_descriptive": time.perf_counter() - start,
            "steps": records, "truncated": False}


def qualify(root, contract, output):
    model, tokenizer, assets = open_model(root, contract)
    traces = []
    for prompt in PROBES:
        ids = encode(tokenizer, prompt, contract["limits"]["prompt_tokens"])
        traces.append(greedy_trace(model, tokenizer, ids, 4, [], True))
    if (inventory(root)["identity"] != assets["identity"]
            or source_identity() != contract["source_identity"]):
        raise ContractError("Checkpoint changed during qualification")
    receipt = {"schema": "qrr.native-receipt/1", "passed": True,
               "evidence_class": "FROZEN_R2_MECHANICAL_ONLY", "contract_identity": digest(contract),
               "source_identity": source_identity(), "model_identity": assets["identity"],
               "python": platform.python_version(), "runtime": contract["runtime"],
               "actual_unique_parameters": sum(p.numel() for p in model.parameters()),
               "probes": traces, "probes_sha256": digest(traces), "licenses_training": False, "licenses_sealed_evaluation": False,
               "qrr_native_qualified": False}
    write_new(output, receipt)
    return receipt


def check_receipt(receipt, contract, model_identity):
    expected = {"schema": "qrr.native-receipt/1", "passed": True,
                "evidence_class": "FROZEN_R2_MECHANICAL_ONLY", "contract_identity": digest(contract),
                "source_identity": source_identity(), "model_identity": model_identity,
                "runtime": contract["runtime"], "licenses_training": False,
                "licenses_sealed_evaluation": False, "qrr_native_qualified": False}
    if (digest({k: receipt.get(k) for k in expected}) != digest(expected)
            or len(receipt.get("probes", [])) != 3
            or receipt.get("probes_sha256") != digest(receipt.get("probes"))):
        raise ContractError("Missing, wrong-scope or stale native receipt")
    for probe in receipt["probes"]:
        steps = probe.get("steps", [])
        if len(steps) != 4 or any(r.get("step") != i or r.get("parity_argmax_equal") is not True
                                  for i, r in enumerate(steps)):
            raise ContractError("Incomplete cache qualification trace")


def run_diagnostics(root, contract, receipt, bank, output_root, run_id):
    validate_contract(contract)
    audit = audit_bank(bank)
    if len(bank["cases"]) > contract["limits"]["cases"]:
        raise ContractError("Case budget exceeded")
    # Reject stale receipt before loading any model; recheck after local inventory.
    check_receipt(receipt, contract, contract["assets"]["identity"])
    model, tokenizer, assets = open_model(root, contract)
    jobs = [(c, arm, encode(tokenizer, prompt_for(c, arm), contract["limits"]["prompt_tokens"]))
            for c in bank["cases"] for arm in ARMS]
    directory = safe_run_dir(output_root, run_id)
    write_new(directory / "STARTED.json", {"contract": digest(contract), "bank": bank["cases_sha256"],
                                          "source": source_identity(), "jobs": len(jobs)})
    rows = []
    for i, (case, arm, ids) in enumerate(jobs):
        generation = greedy_trace(model, tokenizer, ids, contract["limits"]["new_tokens"], contract["eos_token_ids"])
        row = {**score(case, generation["text"]), "case_id": case["case_id"],
               "cluster_id": case["cluster_id"], "op": case["op"], "renderer": case["renderer"],
               "arm": arm, "oracle_assisted": arm != "direct", "split": "open_dev",
               "source_identity": source_identity(), "contract_identity": digest(contract),
               "model_identity": assets["identity"], "bank_identity": bank["cases_sha256"],
               "prompt_sha256": digest(prompt_for(case, arm)), "generation": generation}
        write_new(directory / f"row_{i:05d}.json", row)
        rows.append(row)
    if (inventory(root)["identity"] != assets["identity"]
            or source_identity() != contract["source_identity"]):
        raise ContractError("Checkpoint changed; no completion receipt emitted")
    result = {"schema": "qrr.diagnostics/1", "evidence_class": "EXPLORATORY_FROZEN_R2_ONLY",
              "audit": audit, "source_identity": source_identity(), "contract_identity": digest(contract),
              "rows": rows, "rows_sha256": digest(rows), "training_steps": 0,
              "qrr_installed": False, "licenses_training": False,
              "warning": "Oracle arms are not deployable improvements; fixed order timings are descriptive"}
    write_new(directory / "COMPLETE.json", result)
    return result
