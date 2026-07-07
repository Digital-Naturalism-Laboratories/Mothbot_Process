#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
APPS_DIR="$ROOT_DIR/apps"
DIST_DIR="$APPS_DIR/dist"
RELEASE_DIR="$APPS_DIR/release"

if [ ! -d "$DIST_DIR/Mothbot.app" ]; then
  echo "Missing macOS app bundle at $DIST_DIR/Mothbot.app"
  echo "Run: bash apps/scripts/build_desktop_macos.sh"
  exit 1
fi

VERSION="${MOTHBOT_RELEASE_VERSION:-}"
if [ -z "$VERSION" ]; then
  VERSION="$(python3 -c 'import tomllib, pathlib; p=pathlib.Path("pyproject.toml"); print(tomllib.loads(p.read_text())["project"]["version"])')"
fi
ARCH="$(uname -m)"
TARGET_BASENAME="Mothbot-${VERSION}-macos-${ARCH}"

mkdir -p "$RELEASE_DIR"
rm -f "$RELEASE_DIR/${TARGET_BASENAME}.dmg"

hdiutil create -volname "Mothbot" -srcfolder "$DIST_DIR/Mothbot.app" -ov -format UDZO "$RELEASE_DIR/${TARGET_BASENAME}.dmg"

MAX_BYTES=1932735283
WARN_BYTES=1717986918
DMG="$RELEASE_DIR/${TARGET_BASENAME}.dmg"
file_size_bytes="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).stat().st_size)' "$DMG")"
file_size_human="$(python3 -c 'import pathlib,sys; s=pathlib.Path(sys.argv[1]).stat().st_size; print(f"{s/1024/1024/1024:.2f} GiB")' "$DMG")"
echo "$(basename "$DMG") size: $file_size_human"
if [ "$file_size_bytes" -ge "$WARN_BYTES" ]; then
  echo "::warning::$(basename "$DMG") is above 1.6 GiB ($file_size_human)."
fi
if [ "$file_size_bytes" -ge "$MAX_BYTES" ]; then
  echo "::error::$(basename "$DMG") exceeds 1.8 GiB ($file_size_human)."
  exit 1
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### macOS artifact size"
    echo ""
    echo "| Artifact | Size |"
    echo "| --- | --- |"
    echo "| ${TARGET_BASENAME}.dmg | $file_size_human |"
  } >> "$GITHUB_STEP_SUMMARY"
fi

shasum -a 256 "$DMG" > "$RELEASE_DIR/SHA256SUMS.txt"

echo
echo "Release artifact created:"
ls -lh "$DMG" "$RELEASE_DIR/SHA256SUMS.txt"
