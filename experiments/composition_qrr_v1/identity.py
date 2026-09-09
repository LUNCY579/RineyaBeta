"""Read-only asset identity and fail-closed, content-bound records."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re


class ContractError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ContractError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    def invalid(value):
        raise ContractError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=invalid)


def hash_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"Expected regular non-symlink file: {path}")
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if stamp(before) != stamp(after) or path.is_symlink():
        raise ContractError("File changed during hashing")
    return {"sha256": h.hexdigest(), "bytes": after.st_size}


def inventory(root):
    """Observe bytes only; never execute model code or infer training lineage."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("Checkpoint must be a non-symlink directory")
    root = root.resolve()
    def paths():
        result = []
        for path in root.rglob("*"):
            rel = path.relative_to(root)
            if ".git" in rel.parts:
                continue
            if path.is_symlink():
                raise ContractError(f"Checkpoint contains symlink: {rel}")
            if path.is_file():
                result.append(path)
        return sorted(result)
    selected = paths()
    stamp = lambda p: (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
    before_stamps = {p: stamp(p) for p in selected}
    names = [p.relative_to(root).as_posix() for p in selected]
    if "config.json" not in names or "tokenizer.json" not in names:
        raise ContractError("config.json and tokenizer.json are required")
    weights = [p for p in selected if p.suffix == ".safetensors"]
    if not weights:
        raise ContractError("No safetensors weights; a config is not a checkpoint")
    for path in weights:
        with path.open("rb") as stream:
            prefix = stream.read(128)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ContractError("Unresolved Git LFS weight pointer")
        if path.stat().st_size < 16:
            raise ContractError("Truncated weight file")
    files = {p.relative_to(root).as_posix(): hash_file(p) for p in selected}
    if paths() != selected or any(stamp(p) != before_stamps[p] for p in selected):
        raise ContractError("Checkpoint changed during inventory")
    return {"schema": "qrr.assets/1", "files": files, "identity": digest(files),
            "evidence_class": "BYTE_INVENTORY_ONLY", "lineage_verified": False}


def source_identity():
    root = Path(__file__).resolve().parent
    paths = sorted(root.rglob("*.py")) + [root / "protocol.json"]
    files = {p.relative_to(root).as_posix(): hash_file(p) for p in paths}
    return digest(files)


def write_new(path, record):
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: no historical artifact is silently overwritten.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def safe_run_dir(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ContractError("Unsafe run identifier")
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def positive_int(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ContractError(f"{name} must be an integer in [1,{maximum}]")
    return value
