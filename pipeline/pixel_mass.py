#!/usr/bin/env python3
"""
pixel_mass.py — Background removal and pixel-mass measurement for patches.

Phase 1 — Background removal:
    For each patch, run BiRefNet (via rembg) and save *_nobg.png alongside it.
    Skips patches that already have a _nobg.png (unless overwrite=True).

Phase 2 — Pixel counting:
    Read every *_nobg.png, count non-transparent pixels, convert to mm² using the
    calibration factor, and write pixel_mass_pixels / pixel_mass_mm2 /
    timestamp_pixel_mass back into each shape in the detection JSON.

Calibration is stored per-collection in:
    <dataset_root>/_processed/<rel>/calibration.json
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from core.common import find_detection_matches_processed, current_timestamp
from core.paths import get_processed_folder, resolve_patch_path
from core.preview import emit_preview, clear_preview

_CALIB_FILENAME = "calibration.json"
_ALPHA_THRESHOLD = 50  # pixels with alpha below this (0–255) are treated as background

_rembg_session = None
_rembg_model_name: str | None = None


def _get_session(model_name: str = "birefnet-general-lite"):
    global _rembg_session, _rembg_model_name
    if _rembg_session is not None and _rembg_model_name == model_name:
        return _rembg_session

    print(f"Loading {model_name} background-removal model (first use may download weights)...")
    from rembg import new_session
    import onnxruntime as _ort

    available = _ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("  ⚡ CUDA acceleration active (NVIDIA GPU)")
    elif "DmlExecutionProvider" in available:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        print("  ⚡ DirectML acceleration active (Windows GPU)")
    else:
        providers = ["CPUExecutionProvider"]

    try:
        _rembg_session = new_session(model_name, providers=providers)
    except TypeError:
        _rembg_session = new_session(model_name)

    _rembg_model_name = model_name
    print(f"  {model_name} model ready.")
    return _rembg_session


def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def save_calibration(processed_folder: str, calib: dict) -> None:
    os.makedirs(processed_folder, exist_ok=True)
    path = os.path.join(processed_folder, _CALIB_FILENAME)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(processed_folder: str) -> dict | None:
    path = os.path.join(processed_folder, _CALIB_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-patch helpers
# ---------------------------------------------------------------------------

def _nobg_path(patch_path: str) -> str:
    p = Path(patch_path)
    return str(p.parent / f"{p.stem}_nobg.png")


def _remove_background(patch_path: str, model_name: str = "birefnet-general-lite") -> Image.Image:
    from rembg import remove as rembg_remove
    with Image.open(patch_path) as img:
        img_rgb = img.convert("RGB")
    return rembg_remove(img_rgb, session=_get_session(model_name))


def _count_foreground_pixels(nobg_path: str) -> int:
    with Image.open(nobg_path) as img:
        arr = np.asarray(img.convert("RGBA"))
    return int(np.sum(arr[:, :, 3] >= _ALPHA_THRESHOLD))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    dataset_root: str | None = None,
    pixels_per_mm: float | None = None,
    overwrite_nobg: bool = False,
    overwrite_pixmass: bool = True,
    model_name: str = "birefnet-general-lite",
) -> None:
    """Calculate pixel mass for all patches in *input_path*.

    Parameters
    ----------
    input_path:
        Deployment folder to process.
    dataset_root:
        Top-level folder containing the _processed mirror. Defaults to input_path.
    pixels_per_mm:
        Calibration factor. If None, reads from calibration.json in the processed mirror.
        If neither is available, pixel_mass_mm2 is stored as None.
    overwrite_nobg:
        If False (default), skip patches that already have a _nobg.png on disk.
    overwrite_pixmass:
        If False, skip shapes that already have pixel_mass_pixels in the JSON.
    """
    _dataset_root = dataset_root or input_path

    processed_root = get_processed_folder(input_path, _dataset_root)
    calib = load_calibration(processed_root)
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

    # ── Load all JSONs and collect patch paths ────────────────────────────────
    # json_store: json_path → loaded dict (mutated in Phase 2, written at end)
    json_store = {}
    # all_patches: deduplicated ordered list of patch_abs paths that exist on disk
    all_patches = []
    _seen_patches: set[str] = set()

    for image_path, json_path in bot_pairs:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Cannot read JSON {os.path.basename(json_path)}: {e}")
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        json_store[json_path] = data

        for shape in shapes:
            patch_rel = shape.get("patch_path", "")
            if not patch_rel:
                continue
            patch_abs = resolve_patch_path(patch_rel, image_path, _dataset_root)
            if not os.path.isfile(patch_abs):
                continue
            if patch_abs not in _seen_patches:
                _seen_patches.add(patch_abs)
                all_patches.append(patch_abs)

    if not all_patches:
        print("No patches found — nothing to process.")
        return

    # ── Phase 1: Background Removal ───────────────────────────────────────────
    clear_preview()
    _get_session(model_name)   # load model upfront so ETA reflects only inference time

    to_process = [p for p in all_patches if overwrite_nobg or not os.path.isfile(_nobg_path(p))]
    total_bg = len(to_process)
    skipped_bg = len(all_patches) - total_bg

    if skipped_bg:
        print(f"\n── Phase 1: Background removal — {total_bg} patches ({skipped_bg} already have _nobg.png, skipping)")
    else:
        print(f"\n── Phase 1: Background removal — {total_bg} patches")

    bg_done = 0
    bg_errors = 0
    t_start = time.monotonic()

    for patch_abs in to_process:
        nobg = _nobg_path(patch_abs)
        try:
            rgba = _remove_background(patch_abs, model_name=model_name)
            rgba.save(nobg)
            emit_preview(nobg)
            bg_done += 1
            elapsed = time.monotonic() - t_start
            avg = elapsed / bg_done
            eta_str = _format_eta(avg * (total_bg - bg_done)) if bg_done < total_bg else "done"
            print(f"  ✓ [{bg_done}/{total_bg}] {os.path.basename(patch_abs)} — ETA {eta_str}")
        except Exception as e:
            bg_errors += 1
            print(f"  ❌ [{bg_done + bg_errors}/{total_bg}] {os.path.basename(patch_abs)}: {e}")

    print(f"\n  Phase 1 complete — {bg_done} backgrounds removed, {bg_errors} errors")

    # ── Phase 2: Pixel Counting ───────────────────────────────────────────────
    print(f"\n── Phase 2: Counting foreground pixels and updating JSONs")

    px_done = 0
    px_skipped = 0
    px_errors = 0

    for image_path, json_path in bot_pairs:
        data = json_store.get(json_path)
        if data is None:
            continue

        changed = False
        for shape in data.get("shapes", []):
            if not overwrite_pixmass and "pixel_mass_pixels" in shape:
                px_skipped += 1
                continue

            patch_rel = shape.get("patch_path", "")
            if not patch_rel:
                continue
            patch_abs = resolve_patch_path(patch_rel, image_path, _dataset_root)
            nobg = _nobg_path(patch_abs)

            if not os.path.isfile(nobg):
                continue

            try:
                px_count = _count_foreground_pixels(nobg)
                shape["pixel_mass_pixels"] = px_count
                shape["pixel_mass_mm2"] = (
                    round(px_count / (pixels_per_mm ** 2), 4) if pixels_per_mm else None
                )
                shape["timestamp_pixel_mass"] = current_timestamp()
                changed = True
                px_done += 1

                mm2_str = f"  → {shape['pixel_mass_mm2']:.4f} mm²" if shape["pixel_mass_mm2"] else ""
                print(f"  ✓ {os.path.basename(patch_abs)}: {px_count} px{mm2_str}")
            except Exception as e:
                px_errors += 1
                print(f"  ❌ {os.path.basename(patch_abs)}: {e}")

        if changed:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

    print(f"\n✅ Pixel Mass complete")
    print(f"   Phase 1 (bg removal): {bg_done} done, {skipped_bg} skipped, {bg_errors} errors")
    print(f"   Phase 2 (px count):   {px_done} done, {px_skipped} skipped, {px_errors} errors")
