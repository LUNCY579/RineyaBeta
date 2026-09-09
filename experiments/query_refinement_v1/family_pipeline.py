"""Official dense Rineya successor: fresh qualification -> 25-step smoke -> open DEV.

The original R2 entry point and reports remain historical, unchanged. This path
reuses the exact intervention and training functions, not old evidence.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import torch
from .pipeline import (ARMS, Spec, Runtime, load_protocol, qualify, graph_rows,
                       encode_rows, train_arm, clock, paired_summary)
from experiments.official_family_v1.runtime import (
    FamilyError, SHAPE, binding, check_output, digest, get_profile, hash_file,
    load_model, read_json, require_receipt, seal_receipt, setup_runtime, write_new)

HERE = Path(__file__).resolve().parent


def family_protocol(profile_name):
    overlay = read_json(HERE / "family_protocol.json")
    if (overlay.get("schema") != "rineya.query-family-protocol/1"
            or profile_name not in overlay["profiles"]
            or overlay.get("prompt_protocol") != "raw_completion"
            or overlay.get("evidence_transfer_from_R2") is not False
            or overlay.get("sealed_evaluation_authorized") is not False
            or overlay.get("quality_claim_authorized") is not False):
        raise FamilyError("Invalid official-family query protocol")
    get_profile(profile_name)
    base = load_protocol()  # Retains all original bounded-budget validation.
    for key in ("target_path", "target_config", "branch", "base_commit", "experiment", "schema"):
        base.pop(key, None)
    return {**base, **overlay, "selected_profile": profile_name, "target_config": SHAPE,
            "dtype": "float32", "backend": "eager"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify", "smoke", "dev"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    setup_runtime()
    protocol = family_protocol(args.profile)
    check_output(args.out, args.model, HERE)
    bound = binding(args.model, args.profile, HERE, protocol, args.device)
    if args.stage != "qualify":
        if args.receipt is None:
            raise FamilyError("A fresh official-family predecessor receipt is required")
        require_receipt(read_json(args.receipt), bound,
                        "qualify" if args.stage == "smoke" else "smoke", args.receipt)
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "rineya.family-receipt/1", "stage": args.stage,
               "status": "running", "binding": bound,
               "evidence_class": "PRETRAINED_OFFICIAL_FAMILY_EXPERIMENT",
               "sealed_evaluation_authorized": False, "quality_claim_authorized": False,
               "licenses_next_stage": None}
    model = None
    try:
        model, tokenizer, token_info = load_model(args.model, args.profile, args.device, bound["assets"])
        receipt["tokenizer"] = token_info
        # Native independent oracle, cache, causality, SDPA, enabled-path and gradient checks.
        receipt["qualification"] = qualify(model, protocol["layer"], protocol["rank"])
        if args.stage != "qualify":
            train = graph_rows("train", protocol["train_rows"], protocol["data_seed"])
            dev = graph_rows("dev", protocol["dev_rows"], protocol["data_seed"])
            encoded = encode_rows(tokenizer, train, protocol["max_tokens"])
            encoded_dev = encode_rows(tokenizer, dev, protocol["max_tokens"])
            write_new(args.out / "open_data.json", {"train": train, "dev": dev})
            receipt["data_sha256"] = digest({"train": train, "dev": dev})
            records, results = [], []
            device = torch.device(args.device)
            seeds = protocol["training_seeds"] if args.stage == "dev" else protocol["training_seeds"][:1]
            for seed in seeds:
                for arm in ARMS:
                    torch.manual_seed(seed)
                    rt = Runtime(model, Spec(arm, protocol["layer"], protocol["rank"])).freeze_backbone()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    steps = protocol["dev_steps"] if args.stage == "dev" else protocol["smoke_steps"]
                    metrics = train_arm(rt, encoded, steps, seed, protocol, device)
                    if args.stage == "dev":
                        start = clock(device)
                        for row, prompt, _ in encoded_dev:
                            output = rt.generate(torch.tensor([prompt], device=device),
                                                 max_new_tokens=protocol["max_new_tokens"],
                                                 eos=tuple(token_info["eos_token_ids"]))
                            answer = tokenizer.decode(output, skip_special_tokens=True).strip()
                            records.append({"profile": args.profile, "model_identity": bound["assets"]["identity"],
                                            "arm": arm, "seed": seed, "id": row["id"], "hops": row["hops"],
                                            "prediction": answer, "expected": row["answer"],
                                            "correct": answer == row["answer"], "generated_ids": output})
                        metrics["dev_seconds"] = clock(device) - start
                    metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    if rt.update is not None:
                        torch.save(rt.update.state_dict(), args.out / f"{arm}-{seed}-adapter.pt")
                    results.append({"arm": arm, "seed": seed, **metrics})
                    del rt
            write_new(args.out / "training.json", results)
            receipt["training_budget"] = [{k: row[k] for k in ("arm", "seed", "optimizer_steps", "answer_tokens")}
                                          for row in results]
            if records:
                write_new(args.out / "predictions.json", records)
                write_new(args.out / "summary.json", paired_summary(records))
                write_new(args.out / "by_hop.json", {h: paired_summary([r for r in records if r["hops"] == h]) for h in range(1, 5)})
        if binding(args.model, args.profile, HERE, protocol, args.device) != bound:
            raise FamilyError("Assets/source/runtime changed during execution; no passing receipt")
        receipt["status"] = "pass"
        receipt["licenses_next_stage"] = {"qualify": "smoke", "smoke": "dev", "dev": None}[args.stage]
    except Exception as error:
        receipt.update(status="fail", error=f"{type(error).__name__}: {error}", licenses_next_stage=None)
        raise
    finally:
        receipt["outputs"] = {p.name: hash_file(p) for p in sorted(args.out.iterdir()) if p.is_file()}
        write_new(args.out / "receipt.json", seal_receipt(receipt))
        if model is not None:
            model.zero_grad(set_to_none=True)
    print(f"{args.profile}: {args.stage} passed; sealed evaluation and quality claims remain unauthorized")


if __name__ == "__main__":
    main()
