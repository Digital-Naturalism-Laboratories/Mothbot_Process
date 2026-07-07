#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
APPS_DIR="$ROOT_DIR/apps"
DIST_DIR="$APPS_DIR/dist"
RELEASE_DIR="$APPS_DIR/release"

if [ ! -f "$DIST_DIR/Mothbot/Mothbot" ]; then
  echo "Missing Linux executable at $DIST_DIR/Mothbot/Mothbot"
  echo "Run: bash apps/scripts/build_desktop_linux.sh"
  exit 1
fi

if ! command -v 7z >/dev/null 2>&1; then
  echo "7z is required to package Linux release artifacts."
  echo "Install p7zip-full and rerun."
  exit 1
fi

VERSION="${MOTHBOT_RELEASE_VERSION:-}"
if [ -z "$VERSION" ]; then
  VERSION="$(python3 -c 'import tomllib, pathlib; p=pathlib.Path("pyproject.toml"); print(tomllib.loads(p.read_text())["project"]["version"])')"
fi
ARCH="$(uname -m)"
TARGET_PATH="$RELEASE_DIR/Mothbot-${VERSION}-linux-${ARCH}.zip"

mkdir -p "$RELEASE_DIR"
# Clean up any previous parts before creating new ones
rm -f "${TARGET_PATH}"*

echo "Compressing Linux artifact with 7z split archive (800 MB parts)..."
7z a -tzip -mx=5 -v800m "$TARGET_PATH" "$DIST_DIR/Mothbot"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  total_bytes=0
  rows=""
  for part in "${TARGET_PATH}".*; do
    [ -f "$part" ] || continue
    part_bytes="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).stat().st_size)' "$part")"
    part_human="$(python3 -c 'import pathlib,sys; s=pathlib.Path(sys.argv[1]).stat().st_size; print(f"{s/1024/1024:.0f} MB")' "$part")"
    rows="$rows"$'\n'"| $(basename "$part") | $part_human |"
    total_bytes=$((total_bytes + part_bytes))
  done
  total_human="$(python3 -c "print(f'{${total_bytes}/1024/1024/1024:.2f} GiB')")"
  {
    echo "### Linux artifact size"
    echo ""
    echo "| Part | Size |"
    echo "| --- | --- |"
    echo "$rows"
    echo "| **Total** | **$total_human** |"
  } >> "$GITHUB_STEP_SUMMARY"
fi

echo
echo "Release artifact parts created:"
ls -lh "${TARGET_PATH}".*
