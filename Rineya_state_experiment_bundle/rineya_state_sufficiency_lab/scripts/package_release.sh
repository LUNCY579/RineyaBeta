#!/usr/bin/env bash
# Build a distributable archive of the lab plus a checksum manifest, then verify
# the archive by extracting it into a clean directory and re-running
# compile + tests + the offline reference smoke cycle.
#
# Usage: scripts/package_release.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
OUT_DIR="${1:-dist}"
NAME="rineya_state_sufficiency_lab"
VERSION="$(python -c 'import rineya_lab, sys; sys.stdout.write(rineya_lab.__version__)')"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$OUT_DIR"
PKG_DIR="$STAGE/${NAME}"
mkdir -p "$PKG_DIR"

echo "Staging ${NAME} ${VERSION}..."
for item in src configs docs schemas scripts tests \
            pyproject.toml README.md LICENSE CHANGELOG.md CONTRIBUTING.md SECURITY.md \
            Dockerfile docker-compose.yml Makefile .dockerignore .gitignore .github; do
    [ -e "$item" ] && cp -R "$item" "$PKG_DIR/"
done
find "$PKG_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$PKG_DIR" -name '*.pyc' -delete 2>/dev/null || true
rm -rf "$PKG_DIR/.pytest_cache" "$PKG_DIR/.ruff_cache"

echo "Writing checksum manifest..."
( cd "$PKG_DIR" && find . -type f ! -name 'sha256sums.txt' -print0 | sort -z | \
  while IFS= read -r -d '' f; do
      if command -v sha256sum >/dev/null; then sha256sum "${f#./}";
      else shasum -a 256 "${f#./}"; fi
  done > sha256sums.txt )
FILE_COUNT=$(grep -c . "$PKG_DIR/sha256sums.txt")
echo "  ${FILE_COUNT} files hashed"

ARCHIVE="${REPO_ROOT}/${OUT_DIR}/${NAME}-${VERSION}.zip"
rm -f "$ARCHIVE"
( cd "$STAGE" && zip -q -r "$ARCHIVE" "${NAME}" )
if command -v sha256sum >/dev/null; then sha256sum "$ARCHIVE"; else shasum -a 256 "$ARCHIVE"; fi

echo
echo "Verifying archive integrity..."
python - "$ARCHIVE" <<'PY'
import sys, zipfile
from rineya_lab.data.bundle import inspect_archive
report = inspect_archive(sys.argv[1])
assert zipfile.ZipFile(sys.argv[1]).testzip() is None
print(f"  {report.member_count} members, {report.total_uncompressed_bytes} bytes uncompressed, safety checks OK")
PY

VERIFY_DIR="$STAGE/verify"
mkdir -p "$VERIFY_DIR"
( cd "$VERIFY_DIR" && unzip -q "$ARCHIVE" )
EXTRACTED="$VERIFY_DIR/${NAME}"

echo "Re-verifying file checksums after extraction..."
( cd "$EXTRACTED" && if command -v sha256sum >/dev/null; then sha256sum -c --quiet sha256sums.txt;
  else shasum -a 256 -c sha256sums.txt >/dev/null; fi )
echo "  checksums match"

echo "Compile + tests + reference smoke in the extracted copy..."
( cd "$EXTRACTED" \
  && python -m compileall -q src \
  && PYTHONPATH="$EXTRACTED/src" python -m pytest -q \
  && PYTHONPATH="$EXTRACTED/src" python -m rineya_lab.cli smoke \
       --dataset-root "${RINEYA_DATASET_ROOT:-$REPO_ROOT/../rineya_sequence_independence_dataset_v1}" \
       > /dev/null )

echo
echo "Release verified: $ARCHIVE"
