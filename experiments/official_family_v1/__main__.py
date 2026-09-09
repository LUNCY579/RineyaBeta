"""Explicit asset inventory or authenticated restore; never starts an experiment."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from .runtime import (REQUIRED, OPTIONAL, FamilyError, get_profile, inventory,
                      registry, write_new, check_output, HERE)


def restore(profile_name, destination):
    """Copy only one pinned core from HF cache into a new immutable directory.

    Uses the caller's normal HF authentication; no token is accepted in CLI
    arguments, written into artifacts, or printed. No public-model fallback.
    """
    from huggingface_hub import HfApi, hf_hub_download
    profile, reg = get_profile(profile_name), registry()
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(HERE):
        raise FamilyError("Restore destination must be new and outside experiment source")
    repo, revision = reg["backup_repository"], reg["backup_revision"]
    paths = HfApi().list_repo_files(repo_id=repo, revision=revision, token=True)
    suffix = profile["directory"] + "/config.json"
    matches = [p[:-len("config.json")] for p in paths
               if p in (suffix, "Rineya_Published_Cores/" + suffix)]
    if len(matches) != 1:
        raise FamilyError("Pinned backup must contain exactly one authoritative core directory")
    prefix = matches[0]
    missing = [name for name in REQUIRED if prefix + name not in paths]
    if missing:
        raise FamilyError("Pinned backup is incomplete: " + ", ".join(missing))
    selected = [*REQUIRED, *(n for n in (*OPTIONAL, "LICENSE") if prefix + n in paths)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".rineya-core-", dir=destination.parent))
    try:
        for name in selected:
            cached = hf_hub_download(repo_id=repo, filename=prefix + name,
                                     revision=revision, token=True)
            shutil.copyfile(cached, stage / name)  # Materialize HF-cache symlinks, never execute code.
        assets = inventory(stage, profile_name)
        # Reserve destination exclusively, then move our generated files only.
        destination.mkdir(exist_ok=False)
        try:
            for file in stage.iterdir():
                file.rename(destination / file.name)
        except Exception:
            # Leave partial output visibly incomplete; never overwrite an existing core.
            raise
        return {"directory": str(destination), "repository": repo, "revision": revision,
                "assets": assets, "pretrained_model_executed": False, "licenses_training": False}
    finally:
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "restore"))
    parser.add_argument("--profile", required=True, choices=tuple(registry()["profiles"]))
    parser.add_argument("--model", type=Path, help="Existing materialized core (inventory)")
    parser.add_argument("--destination", type=Path, help="New directory (restore)")
    parser.add_argument("--out", type=Path, help="Optional new inventory receipt outside the checkpoint")
    args = parser.parse_args()
    root = args.model if args.action == "inventory" else args.destination
    if root is None:
        parser.error("inventory requires --model; restore requires --destination")
    if args.out:
        check_output(args.out, root, HERE)
        if args.out.exists():
            parser.error("--out must not already exist")
    result = inventory(root, args.profile) if args.action == "inventory" else restore(args.profile, root)
    if args.out:
        write_new(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
