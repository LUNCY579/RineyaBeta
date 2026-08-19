from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rineya_lab.data.bundle import (
    BundleError,
    extract_bundle,
    inspect_archive,
    parse_sha256sums,
    verify_checksums,
)
from rineya_lab.utils.hashing import sha256_file


def make_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def test_extracts_a_normal_archive(tmp_path):
    archive = make_zip(tmp_path / "ok.zip", {"data/a.txt": "hello", "README.md": "hi"})
    destination = tmp_path / "out"
    report = extract_bundle(archive, destination)
    assert report.extracted and report.member_count == 2
    assert (destination / "data" / "a.txt").read_text() == "hello"


def test_rejects_parent_directory_traversal(tmp_path):
    archive = make_zip(tmp_path / "evil.zip", {"../escape.txt": "nope"})
    with pytest.raises(BundleError, match="traversal"):
        inspect_archive(archive)


def test_rejects_absolute_paths(tmp_path):
    archive = tmp_path / "abs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/etc/passwd", "nope")
    with pytest.raises(BundleError, match="Absolute path"):
        inspect_archive(archive)


def test_rejects_symlink_members(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(BundleError, match="Symlink"):
        inspect_archive(archive)


def test_rejects_zip_bomb_ratio(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.txt", "A" * (5 * 1024 * 1024))
    with pytest.raises(BundleError, match="compression ratio"):
        inspect_archive(archive)


def test_rejects_non_zip(tmp_path):
    path = tmp_path / "not.zip"
    path.write_text("plain text")
    with pytest.raises(BundleError, match="valid ZIP"):
        inspect_archive(path)


def test_missing_archive(tmp_path):
    with pytest.raises(BundleError, match="not found"):
        inspect_archive(tmp_path / "absent.zip")


def test_reuses_a_populated_destination(tmp_path):
    archive = make_zip(tmp_path / "ok.zip", {"a.txt": "one"})
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "sentinel").write_text("keep me")
    report = extract_bundle(archive, destination)
    assert not report.extracted
    assert (destination / "sentinel").exists()


def test_checksum_verification_round_trip(tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    target = root / "data" / "file.txt"
    target.write_text("payload")
    sums = root / "sha256sums.txt"
    sums.write_text(f"{sha256_file(target)}  data/file.txt\n")
    report = verify_checksums(root, sums)
    assert report.ok and report.verified_files == 1


def test_checksum_mismatch_is_reported(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "file.txt"
    target.write_text("payload")
    sums = root / "sums.txt"
    sums.write_text("0" * 64 + "  file.txt\n")
    report = verify_checksums(root, sums)
    assert not report.ok and report.hash_mismatches


def test_parse_sha256sums_handles_star_prefix_and_comments(tmp_path):
    path = tmp_path / "sums.txt"
    path.write_text("# comment\n" + "a" * 64 + "  *bin.dat\n\n")
    assert parse_sha256sums(path) == {"bin.dat": "a" * 64}


def test_real_dataset_checksums_verify(dataset_root):
    sums = dataset_root / "data" / "manifests" / "sha256sums.txt"
    if not sums.exists():
        import pytest as _pytest

        _pytest.skip("dataset has no sha256sums.txt")
    report = verify_checksums(dataset_root, sums, strict=False)
    assert not report.hash_mismatches, report.hash_mismatches[:3]
    assert report.verified_files > 0
