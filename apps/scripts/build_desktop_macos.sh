#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
APPS_DIR="$ROOT_DIR/apps"
BUILD_DIR="$APPS_DIR/build"
DIST_DIR="$APPS_DIR/dist"

mkdir -p "$BUILD_DIR" "$DIST_DIR"

VENV_DIR="$ROOT_DIR/.venv-packaging"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export ULTRALYTICS_AUTOINSTALL=0
if [[ "${GITHUB_REF_NAME:-}" =~ ^v.+$ ]]; then
  export MOTHBOT_RELEASE_VERSION="${GITHUB_REF_NAME#v}"
else
  export MOTHBOT_RELEASE_VERSION="$(python3 -c 'import tomllib, pathlib; p=pathlib.Path("pyproject.toml"); print(tomllib.loads(p.read_text())["project"]["version"])')"
fi

printf "%s\n" "$MOTHBOT_RELEASE_VERSION" > "$BUILD_DIR/VERSION"
export MOTHBOT_VERSION_FILE="$BUILD_DIR/VERSION"

python -m pip install --upgrade pip
python -m pip install -e ".[cpu,packaging]"

# Generate a proper macOS .icns from favicon.png (required for Dock/Finder icon).
# sips and iconutil are standard macOS tools, always available on the build runner.
FAVICON_PNG="$ROOT_DIR/assets/favicon.png"
ICNS_OUT="$ROOT_DIR/assets/mothbot.icns"
ICONSET_TMP="$(mktemp -d)/mothbot.iconset"
mkdir -p "$ICONSET_TMP"
for size in 16 32 128 256 512; do
    sips -z $size $size "$FAVICON_PNG" --out "$ICONSET_TMP/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$FAVICON_PNG" --out "$ICONSET_TMP/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET_TMP" -o "$ICNS_OUT"
rm -rf "$(dirname "$ICONSET_TMP")"
echo "Generated $ICNS_OUT"

# Fetch large models (e.g. the default birefnet bg-removal model) into assets/
# so the PyInstaller spec can bundle them into the app.
python apps/scripts/fetch_bundled_models.py

python -m PyInstaller --clean --noconfirm \
  --workpath "$BUILD_DIR" \
  --distpath "$DIST_DIR" \
  apps/packaging/pyinstaller/mothbot_desktop.spec

bash "./apps/scripts/package_release_macos.sh"

echo
echo "Build complete."
echo "Artifact: $DIST_DIR/Mothbot.app"
