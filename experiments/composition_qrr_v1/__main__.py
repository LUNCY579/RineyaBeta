"""Bounded entrypoints. Training and sealed evaluation deliberately have no command."""
import argparse
import json
from pathlib import Path
from .identity import ContractError, inventory, read_json, source_identity, write_new
from .instrument import audit_bank, build_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-id")
    commands.add_parser("arms")
    inv = commands.add_parser("inventory")
    inv.add_argument("checkpoint", type=Path)
    inv.add_argument("--out", required=True, type=Path)
    bank = commands.add_parser("build-bank")
    bank.add_argument("--worlds", type=int, default=8)
    bank.add_argument("--seed", type=int, default=17)
    bank.add_argument("--padding-lines", type=int, default=0)
    bank.add_argument("--placement", choices=("start", "middle", "end"), default="middle")
    bank.add_argument("--out", required=True, type=Path)
    audit = commands.add_parser("audit-bank")
    audit.add_argument("bank", type=Path)
    for name in ("qualify-native", "diagnose-native"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--contract", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "diagnose-native":
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--bank", type=Path, required=True)
            p.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "source-id":
            result = {"source_identity": source_identity()}
        elif args.command == "arms":
            from .qrr import arm_table
            result = arm_table()
        elif args.command == "inventory":
            result = inventory(args.checkpoint)
            write_new(args.out, result)
        elif args.command == "build-bank":
            result = build_bank(args.worlds, args.seed, args.padding_lines, args.placement)
            audit_bank(result)
            write_new(args.out, result)
            result = {"path": str(args.out), "cases_sha256": result["cases_sha256"]}
        elif args.command == "audit-bank":
            result = audit_bank(read_json(args.bank))
        else:
            from .native import qualify, run_diagnostics
            contract = read_json(args.contract)
            if args.command == "qualify-native":
                result = qualify(args.checkpoint, contract, args.out)
            else:
                result = run_diagnostics(args.checkpoint, contract, read_json(args.receipt),
                                         read_json(args.bank), args.out, args.run_id)
                result = {k: v for k, v in result.items() if k != "rows"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")


if __name__ == "__main__":
    main()
