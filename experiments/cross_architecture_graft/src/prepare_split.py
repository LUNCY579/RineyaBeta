"""CLI for deterministic source-family train/dev splitting."""

from __future__ import annotations

import argparse
import json

from .splits import build_grouped_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--dev-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--group-fields",
        default="source_family,template_id,generator_id,group_id",
        help="comma-separated precedence list; a row without all fields is rejected",
    )
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    record = build_grouped_split(
        input_path=args.input,
        train_path=args.train_out,
        dev_path=args.dev_out,
        manifest_path=args.manifest_out,
        group_fields=[field.strip() for field in args.group_fields.split(",") if field.strip()],
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        category_field=args.category_field,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
