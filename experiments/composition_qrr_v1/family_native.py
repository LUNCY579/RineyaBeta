"""Official Rineya dense-core frozen diagnostics; native QRR/training stay blocked."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
from .native import PROBES, encode, greedy_trace
from .instrument import ARMS, audit_bank, prompt_for, score
from experiments.official_family_v1.runtime import (
    FamilyError, binding, check_output, digest, get_profile, hash_file, load_model,
    read_json, require_receipt, seal_receipt, setup_runtime, write_new)

HERE = Path(__file__).resolve().parent


def family_protocol(profile_name):
    protocol = read_json(HERE / "family_protocol.json")
    if (protocol.get("schema") != "rineya.qrr-family-protocol/1"
            or profile_name not in protocol["profiles"]
            or protocol.get("prompt_protocol") != "raw_completion"
            or protocol.get("limits") != {"prompt_tokens": 4096, "new_tokens": 24, "cases": 768}
            or protocol.get("dtype") != "float32" or protocol.get("backend") != "eager"
            or any(protocol.get(key) is not False for key in
                   ("licenses_training", "qrr_native_qualified", "sealed_evaluation_authorized",
                    "quality_claim_authorized", "evidence_transfer_from_R2"))):
        raise FamilyError("Unregistered official-family frozen protocol")
    get_profile(profile_name)
    return {**protocol, "selected_profile": profile_name}


def validate_contract(contract, bound):
    if (contract.get("schema") != "rineya.family-frozen-contract/1"
            or contract.get("status") != "REVIEWED_FROZEN_ONLY"
            or not isinstance(contract.get("reviewed_by"), str) or not contract["reviewed_by"].strip()
            or contract.get("binding") != bound
            or contract.get("licenses_training") is not False
            or contract.get("qrr_native_qualified") is not False):
        raise FamilyError("A fresh, independently reviewed, source-bound family frozen contract is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "qualify", "diagnose"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    setup_runtime()
    protocol = family_protocol(args.profile)
    check_output(args.out, args.model, HERE)
    bound = binding(args.model, args.profile, HERE, protocol, args.device)
    if args.stage == "prepare":
        write_new(args.out, {"schema": "rineya.family-frozen-contract/1", "status": "UNREVIEWED",
                             "reviewed_by": "", "binding": bound, "licenses_training": False,
                             "qrr_native_qualified": False,
                             "review_instruction": "Review exact core, tokenizer, runtime, raw-completion protocol and limits; then set REVIEWED_FROZEN_ONLY and your reviewer identity. This draft is not a qualification."})
        return
    if args.contract is None:
        raise FamilyError("--contract is required; asset inventory cannot authorize execution")
    contract = read_json(args.contract)
    validate_contract(contract, bound)
    if args.stage == "diagnose":
        if args.receipt is None or args.bank is None:
            raise FamilyError("--receipt and --bank are required for open diagnostics")
        previous = read_json(args.receipt)
        require_receipt(previous, bound, "qualify", args.receipt)
        if previous.get("contract_identity") != digest(contract):
            raise FamilyError("Receipt belongs to a different reviewed contract")
        bank = read_json(args.bank)
        audit = audit_bank(bank)
        if len(bank["cases"]) > protocol["limits"]["cases"]:
            raise FamilyError("Case budget exceeded")
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "rineya.family-receipt/1", "stage": args.stage, "status": "running",
               "binding": bound, "contract_identity": digest(contract),
               "evidence_class": "PRETRAINED_OFFICIAL_FAMILY_EXPERIMENT",
               "licenses_training": False, "qrr_native_qualified": False,
               "sealed_evaluation_authorized": False, "quality_claim_authorized": False,
               "licenses_next_stage": None}
    try:
        model, tokenizer, token_info = load_model(args.model, args.profile, args.device, bound["assets"])
        receipt["tokenizer"] = token_info
        limits = protocol["limits"]
        receipt["qualification"] = [greedy_trace(model, tokenizer, encode(tokenizer, p, limits["prompt_tokens"]),
                                                    4, [], True) for p in PROBES]
        if args.stage == "diagnose":
            # Pre-encode all jobs to fail before partial results if a prompt exceeds budget.
            jobs = [(case, arm, encode(tokenizer, prompt_for(case, arm), limits["prompt_tokens"]))
                    for case in bank["cases"] for arm in ARMS]
            rows = []
            for i, (case, arm, ids) in enumerate(jobs):
                generation = greedy_trace(model, tokenizer, ids, limits["new_tokens"], token_info["eos_token_ids"])
                row = {**score(case, generation["text"]), "case_id": case["case_id"],
                       "cluster_id": case["cluster_id"], "op": case["op"], "renderer": case["renderer"],
                       "arm": arm, "oracle_assisted": arm != "direct", "split": "open_dev",
                       "profile": args.profile, "core_id": bound["assets"]["profile"]["core_id"],
                       "model_identity": bound["assets"]["identity"], "binding_identity": digest(bound),
                       "bank_identity": bank["cases_sha256"],
                       "prompt_sha256": digest(prompt_for(case, arm)), "generation": generation}
                write_new(args.out / f"row_{i:05d}.json", row)
                rows.append(row)
            write_new(args.out / "diagnostics.json", {"schema": "rineya.family-qrr-diagnostics/1",
                       "audit": audit, "rows": rows, "rows_sha256": digest(rows),
                       "training_steps": 0, "qrr_installed": False,
                       "warning": "Oracle-assisted arms are not deployable improvements; fixed-order timings are descriptive."})
        if binding(args.model, args.profile, HERE, protocol, args.device) != bound:
            raise FamilyError("Assets/source/runtime changed; no passing receipt")
        receipt["status"] = "pass"
        receipt["licenses_next_stage"] = "diagnose" if args.stage == "qualify" else None
    except Exception as error:
        receipt.update(status="fail", error=f"{type(error).__name__}: {error}", licenses_next_stage=None)
        raise
    finally:
        receipt["outputs"] = {p.name: hash_file(p) for p in sorted(args.out.iterdir()) if p.is_file()}
        write_new(args.out / "receipt.json", seal_receipt(receipt))
    print(f"{args.profile}: {args.stage} passed; native QRR, optimization and sealed evaluation remain blocked")


if __name__ == "__main__":
    main()
