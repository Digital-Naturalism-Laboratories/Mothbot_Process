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

VERSION="$(python3 -c 'import tomllib, pathlib; p=pathlib.Path("pyproject.toml"); print(tomllib.loads(p.read_text())["project"]["version"])')"
ARCH="$(uname -m)"
TARGET_PATH="$RELEASE_DIR/Mothbot-${VERSION}-linux-${ARCH}.zip"

mkdir -p "$RELEASE_DIR"
rm -f "$TARGET_PATH"

# Keep .zip output for user compatibility but use 7z compression for better ratios.
7z a -tzip -mx=9 "$TARGET_PATH" "$DIST_DIR/Mothbot" >/dev/null

MAX_BYTES=1932735283
WARN_BYTES=1717986918
FILE_SIZE_BYTES="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).stat().st_size)' "$TARGET_PATH")"
FILE_SIZE_HUMAN="$(python3 -c 'import pathlib,sys; s=pathlib.Path(sys.argv[1]).stat().st_size; print(f"{s/1024/1024/1024:.2f} GiB")' "$TARGET_PATH")"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Linux artifact size"
    echo ""
    echo "| Artifact | Size |"
    echo "| --- | --- |"
    echo "| $(basename "$TARGET_PATH") | $FILE_SIZE_HUMAN |"
  } >> "$GITHUB_STEP_SUMMARY"
fi

echo "Linux release artifact size: $FILE_SIZE_HUMAN"
if [ "$FILE_SIZE_BYTES" -ge "$WARN_BYTES" ]; then
  echo "::warning::Linux release artifact is above 1.6 GiB ($FILE_SIZE_HUMAN)."
fi
if [ "$FILE_SIZE_BYTES" -ge "$MAX_BYTES" ]; then
  echo "::error::Linux release artifact exceeds 1.8 GiB ($FILE_SIZE_HUMAN)."
  exit 1
fi

echo
echo "Release artifact created:"
ls -lh "$TARGET_PATH"
