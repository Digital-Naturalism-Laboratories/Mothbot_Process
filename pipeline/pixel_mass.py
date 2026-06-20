#!/usr/bin/env python3
"""
pixel_mass.py — Background removal and pixel-mass measurement for patches.

For each detected patch:
1. Load the patch from the _processed mirror
2. Remove background with BiRefNet (via rembg) → save *_nobg.png alongside the patch
3. Count non-transparent pixels → pixel_mass_pixels
4. Convert to physical area using pixels_per_mm calibration → pixel_mass_mm2
5. Write pixel_mass_pixels / pixel_mass_mm2 / timestamp_pixel_mass back into each
   shape in the detection JSON

Calibration is stored per-collection in:
    <dataset_root>/_processed/<rel>/calibration.json
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from core.common import find_detection_matches_processed, current_timestamp
from core.paths import get_processed_folder, resolve_patch_path

_CALIB_FILENAME = "calibration.json"

# Module-level BiRefNet session — loaded once on first use.
_rembg_session = None


def _get_session():
    global _rembg_session
    if _rembg_session is None:
        print("Loading BiRefNet background-removal model (first use — may download ~175 MB)...")
        from rembg import new_session
        _rembg_session = new_session("birefnet-general")
        print("BiRefNet model ready.")
    return _rembg_session


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def save_calibration(processed_folder: str, calib: dict) -> None:
    """Write *calib* dict to *processed_folder*/calibration.json."""
    os.makedirs(processed_folder, exist_ok=True)
    path = os.path.join(processed_folder, _CALIB_FILENAME)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(processed_folder: str) -> dict | None:
    """Return calibration dict from *processed_folder*/calibration.json, or None."""
    path = os.path.join(processed_folder, _CALIB_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-patch processing
# ---------------------------------------------------------------------------

def _remove_background(patch_path: str) -> Image.Image:
    """Return an RGBA PIL Image with background removed via BiRefNet."""
    from rembg import remove as rembg_remove
    with Image.open(patch_path) as img:
        img_rgb = img.convert("RGB")
    return rembg_remove(img_rgb, session=_get_session())


def _count_foreground_pixels(rgba: Image.Image) -> int:
    """Count non-transparent pixels (alpha > 0) in an RGBA image."""
    arr = np.asarray(rgba)
    return int(np.sum(arr[:, :, 3] > 0))


def _nobg_path(patch_path: str) -> str:
    """Return the *_nobg.png path alongside *patch_path*."""
    p = Path(patch_path)
    return str(p.parent / f"{p.stem}_nobg.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    dataset_root: str | None = None,
    pixels_per_mm: float | None = None,
    overwrite: bool = False,
) -> None:
    """Calculate pixel mass for all patches in *input_path*.

    Parameters
    ----------
    input_path:
        Deployment folder to process (same convention as all other pipeline scripts).
    dataset_root:
        Top-level folder that contains the _processed mirror.  Defaults to input_path.
    pixels_per_mm:
        Calibration factor.  If None, reads from calibration.json in the processed
        mirror for this collection.  If neither is available, pixel_mass_mm2 is stored
        as None but pixel_mass_pixels is still calculated.
    overwrite:
        If False, shapes that already have pixel_mass_pixels are skipped.
    """
    _dataset_root = dataset_root or input_path

    processed_root_for_input = get_processed_folder(input_path, _dataset_root)
    calib = load_calibration(processed_root_for_input)
    if pixels_per_mm is None and calib:
        pixels_per_mm = calib.get("pixels_per_mm")

    if pixels_per_mm:
        print(f"  Calibration: {pixels_per_mm:.4f} px/mm")
    else:
        print("⚠️  No calibration found — pixel_mass_mm2 will be None.")
        print("   Use the Pixel Mass tab to set calibration first.")

    _hu_pairs, bot_pairs = find_detection_matches_processed(_dataset_root, source_folder=input_path)

    if not bot_pairs:
        print("No detection JSON files found — nothing to process.")
        return

    total = len(bot_pairs)
    processed = 0
    skipped = 0
    errors = 0

    for i, (image_path, json_path) in enumerate(bot_pairs):
        print(f"[{i + 1}/{total}] {os.path.basename(image_path)}")

        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Cannot read JSON: {e}")
            errors += 1
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            print("  · no detections, skipping")
            continue

        changed = False
        for shape in shapes:
            if not overwrite and "pixel_mass_pixels" in shape:
                skipped += 1
                continue

            patch_rel = shape.get("patch_path", "")
            if not patch_rel:
                print("  ⚠️  shape missing patch_path, skipping")
                continue

            patch_abs = resolve_patch_path(patch_rel, image_path, _dataset_root)
            if not os.path.isfile(patch_abs):
                print(f"  ⚠️  patch not found: {os.path.basename(patch_abs)}")
                continue

            try:
                rgba = _remove_background(patch_abs)
                px_count = _count_foreground_pixels(rgba)
                rgba.save(_nobg_path(patch_abs))

                shape["pixel_mass_pixels"] = px_count
                shape["pixel_mass_mm2"] = (
                    round(px_count / (pixels_per_mm ** 2), 4) if pixels_per_mm else None
                )
                shape["timestamp_pixel_mass"] = current_timestamp()
                changed = True
                processed += 1

                mm2_str = f"  → {shape['pixel_mass_mm2']:.4f} mm²" if shape["pixel_mass_mm2"] else ""
                print(f"  ✓ {os.path.basename(patch_abs)}: {px_count} px{mm2_str}")

            except Exception as e:
                print(f"  ❌ {os.path.basename(patch_abs)}: {e}")
                errors += 1

        if changed:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

    print(f"\n✅ Pixel Mass complete — {processed} patches processed, {skipped} skipped, {errors} errors")
