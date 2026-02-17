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

VERSION="$(python3 -c 'import tomllib, pathlib; p=pathlib.Path("pyproject.toml"); print(tomllib.loads(p.read_text())["project"]["version"])')"
ARCH="$(uname -m)"
TARGET_BASENAME="Mothbot-${VERSION}-macos-${ARCH}"

mkdir -p "$RELEASE_DIR"
rm -f "$RELEASE_DIR/${TARGET_BASENAME}.zip" "$RELEASE_DIR/${TARGET_BASENAME}.dmg"

ditto -c -k --sequesterRsrc --keepParent "$DIST_DIR/Mothbot.app" "$RELEASE_DIR/${TARGET_BASENAME}.zip"
hdiutil create -volname "Mothbot" -srcfolder "$DIST_DIR/Mothbot.app" -ov -format UDZO "$RELEASE_DIR/${TARGET_BASENAME}.dmg"

shasum -a 256 \
  "$RELEASE_DIR/${TARGET_BASENAME}.zip" \
  "$RELEASE_DIR/${TARGET_BASENAME}.dmg" > "$RELEASE_DIR/SHA256SUMS.txt"

echo
echo "Release artifacts created:"
ls -lh "$RELEASE_DIR/${TARGET_BASENAME}.zip" "$RELEASE_DIR/${TARGET_BASENAME}.dmg" "$RELEASE_DIR/SHA256SUMS.txt"
