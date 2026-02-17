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

python -m pip install --upgrade pip
python -m pip install -e ".[cpu,packaging]"
python -m PyInstaller --clean --noconfirm \
  --workpath "$BUILD_DIR" \
  --distpath "$DIST_DIR" \
  apps/packaging/pyinstaller/mothbot_desktop.spec

echo
echo "Build complete."
echo "Artifact: $DIST_DIR/Mothbot"
