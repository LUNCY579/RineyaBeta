"""Verify CAG evidence payload hashes and claim-boundary invariants."""

from __future__ import annotations

import argparse
import json

from .evidence import load_verified_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()
    verified = []
    for path in args.manifests:
        manifest = load_verified_manifest(path)
        verified.append(
            {
                "path": path,
                "experiment_id": manifest["experiment_id"],
                "phase": manifest["phase"],
                "status": manifest["status"],
                "sealed_evaluation_opened": manifest["sealed_evaluation_opened"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            }
        )
    print(json.dumps({"status": "PASS", "verified": verified}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
