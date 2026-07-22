#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
APPS_DIR="$ROOT_DIR/apps"
BUILD_DIR="$APPS_DIR/build"
DIST_DIR="$APPS_DIR/dist"

# Free pre-installed tooling that Mothbot never needs (~14 GB on ubuntu-latest).
echo "Freeing pre-installed runner disk space..."
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL 2>/dev/null || true
df -h . || true

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

# Fetch large models (e.g. the default birefnet bg-removal model) into assets/
# so the PyInstaller spec can bundle them into the app.
python apps/scripts/fetch_bundled_models.py

python -m PyInstaller --clean --noconfirm \
  --workpath "$BUILD_DIR" \
  --distpath "$DIST_DIR" \
  apps/packaging/pyinstaller/mothbot_desktop.spec

# Free disk space before compression — the venv and PyInstaller work dir are
# no longer needed after the build completes, but together they occupy ~5+ GB
# that 7z needs to write the compressed split-archive parts.
echo "Freeing disk space before compression..."
deactivate 2>/dev/null || true
rm -rf "$VENV_DIR"
rm -rf "$BUILD_DIR"
df -h . || true  # log remaining space for CI debugging

bash "./apps/scripts/package_release_linux.sh"

echo
echo "Build complete."
echo "Artifact: $DIST_DIR/Mothbot"
