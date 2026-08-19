"""Safe handling of a bundled dataset ZIP.

The archive is untrusted input as far as this code is concerned, so extraction
defends against:

* zip-slip (``../`` members and absolute paths escaping the destination);
* symlink members, which can redirect a later write outside the destination;
* implausible member counts and decompressed sizes (zip bombs).

Integrity is checked against ``data/manifests/sha256sums.txt`` when present.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rineya_lab.utils.hashing import sha256_file
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("data.bundle")

MAX_MEMBERS = 20_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 8 * 1024**3  # 8 GiB
# Deflate tops out near 1032:1 on uniform bytes; real JSONL compresses ~5-15x.
# 500 leaves ample headroom for legitimate data while catching bomb-shaped members.
MAX_COMPRESSION_RATIO = 500


class BundleError(RuntimeError):
    """Raised when an archive fails an integrity or safety check."""


@dataclass
class BundleReport:
    archive: str
    destination: str
    member_count: int = 0
    total_uncompressed_bytes: int = 0
    extracted: bool = False
    verified_files: int = 0
    hash_mismatches: list[str] = field(default_factory=list)
    missing_from_manifest: list[str] = field(default_factory=list)
    ok: bool = False


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def inspect_archive(archive: str | Path) -> BundleReport:
    """Validate archive structure without writing anything to disk."""
    archive = Path(archive)
    report = BundleReport(archive=str(archive), destination="")
    if not archive.exists():
        raise BundleError(f"Bundle not found: {archive}")
    if not zipfile.is_zipfile(archive):
        raise BundleError(f"Not a valid ZIP archive: {archive}")

    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise BundleError(f"Corrupt member in {archive}: {bad}")
        infos = zf.infolist()
        report.member_count = len(infos)
        if report.member_count > MAX_MEMBERS:
            raise BundleError(f"Archive has {report.member_count} members; refusing (limit {MAX_MEMBERS})")
        total = 0
        for info in infos:
            name = info.filename
            if name.startswith("/") or Path(name).is_absolute():
                raise BundleError(f"Absolute path member rejected: {name}")
            if ".." in Path(name).parts:
                raise BundleError(f"Parent-directory traversal member rejected: {name}")
            # Unix mode is stored in the top 16 bits of external_attr; 0o120000 == symlink.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise BundleError(f"Symlink member rejected: {name}")
            total += info.file_size
            if info.compress_size > 0 and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
                raise BundleError(f"Suspicious compression ratio for {name}; refusing to extract")
        report.total_uncompressed_bytes = total
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise BundleError(
                f"Archive expands to {total} bytes, above the {MAX_TOTAL_UNCOMPRESSED_BYTES} byte guard"
            )
    report.ok = True
    return report


def extract_bundle(archive: str | Path, destination: str | Path, *, overwrite: bool = False) -> BundleReport:
    """Safely extract ``archive`` into ``destination``."""
    archive = Path(archive)
    destination = Path(destination)
    report = inspect_archive(archive)
    report.destination = str(destination)

    if destination.exists() and any(destination.iterdir()) and not overwrite:
        LOGGER.info("Destination %s is non-empty; reusing existing extraction", destination)
        report.extracted = False
        return report

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = destination / info.filename
            if not _is_within(destination, target if info.is_dir() else target.parent):
                raise BundleError(f"Member escapes destination after join: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
    report.extracted = True
    LOGGER.info("Extracted %d members from %s into %s", report.member_count, archive, destination)
    return report


def parse_sha256sums(path: str | Path) -> dict[str, str]:
    """Parse a ``sha256sums.txt`` file (``<hex>  <relative path>`` per line)."""
    entries: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts
        entries[name.lstrip("*").strip()] = digest.lower()
    return entries


def verify_checksums(root: str | Path, sums_file: str | Path, *, strict: bool = True) -> BundleReport:
    """Verify every file listed in ``sums_file`` relative to ``root``."""
    root = Path(root)
    report = BundleReport(archive=str(sums_file), destination=str(root))
    entries = parse_sha256sums(sums_file)
    for relative, expected in sorted(entries.items()):
        candidates = [root / relative, root / Path(relative).name]
        # sha256sums.txt in the released dataset is written from data/manifests/,
        # so entries may be relative to the dataset root or to the manifest dir.
        candidates.append(root / "data" / "manifests" / Path(relative).name)
        target = next((c for c in candidates if c.exists()), None)
        if target is None:
            report.missing_from_manifest.append(relative)
            continue
        actual = sha256_file(target)
        report.verified_files += 1
        if actual != expected:
            report.hash_mismatches.append(f"{relative}: expected {expected}, got {actual}")
    report.ok = not report.hash_mismatches and (not strict or not report.missing_from_manifest)
    if report.hash_mismatches:
        LOGGER.error("Checksum mismatches: %s", report.hash_mismatches[:5])
    if report.missing_from_manifest:
        LOGGER.warning("Files listed in the manifest were not found: %s", report.missing_from_manifest[:5])
    return report
