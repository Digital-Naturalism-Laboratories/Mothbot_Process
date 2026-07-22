#!/usr/bin/env python3
"""Fetch large model weights that are too big for git into assets/ before packaging.

Some models the desktop app relies on exceed GitHub's 100 MB per-file limit, so
they can't be committed like the (smaller) DINOv2 weights. This script downloads
them into assets/ at build time; the PyInstaller spec then bundles whatever it
finds there. Called by every build script (macOS/Linux/Windows) after deps are
installed and before PyInstaller runs.

Idempotent: skips a model that already exists in assets/ with the right md5, and
prefers copying from a local rembg cache (~/.u2net or $U2NET_HOME) over a network
download so repeat/local builds are fast.

Currently bundles:
  - birefnet-general-lite.onnx  (~224 MB) — default background-removal model, so
    the packaged app never has to download it on first use.
"""
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets"

MODELS = [
    {
        "dest": "birefnet-general-lite.onnx",
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "md5": "4fab47adc4ff364be1713e97b7e66334",
    },
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _u2net_home() -> Path:
    return Path(os.path.expanduser(
        os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net"))
    ))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        dest = ASSETS / m["dest"]

        if dest.exists() and _md5(dest) == m["md5"]:
            print(f"✓ {m['dest']} already in assets/ — skipping")
            continue

        # Prefer an already-downloaded copy from the local rembg cache.
        cached = _u2net_home() / m["dest"]
        if cached.is_file() and _md5(cached) == m["md5"]:
            print(f"↳ Copying {m['dest']} from local cache {cached}")
            shutil.copy2(cached, dest)
            continue

        print(f"↓ Downloading {m['dest']} from {m['url']}")
        urllib.request.urlretrieve(m["url"], dest)
        got = _md5(dest)
        if got != m["md5"]:
            dest.unlink(missing_ok=True)
            sys.exit(f"❌ checksum mismatch for {m['dest']}: got {got}, expected {m['md5']}")
        print(f"✓ {m['dest']} downloaded and verified")


if __name__ == "__main__":
    main()
