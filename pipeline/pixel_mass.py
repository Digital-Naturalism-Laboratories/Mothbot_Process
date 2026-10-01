#!/usr/bin/env python3
"""
pixel_mass.py — Background removal and pixel-mass measurement for patches.

Phase 1 — Background removal:
    For each patch, run BiRefNet (via rembg) or the quick border-colour mask
    (core/colour_mask.py) and save *_nobg.png alongside it. With hybrid
    optimization on, small or blurry patches use the colour mask and the rest
    use the chosen model. Each _nobg.png records the method that made it.
    Skips patches that already have a _nobg.png (unless overwrite=True), except
    colour-mask results are redone when the model is now asked for.

Phase 2 — Pixel counting:
    Read every *_nobg.png, count non-transparent pixels, convert to mm² using the
    calibration factor, and write pixel_mass_pixels / pixel_mass_mm2 /
    timestamp_pixel_mass back into each shape in the detection JSON.

Calibration is stored per-collection in:
    <dataset_root>/_processed/<rel>/calibration.json
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from core.blur import fill_missing_blur_scores
from core.colour_mask import COLOUR_MASK_METHOD, insect_mask
from core.common import find_detection_matches_processed, current_timestamp
from core.paths import get_processed_folder, resolve_patch_path
from core.preview import emit_preview, clear_preview

_CALIB_FILENAME = "calibration.json"
_ALPHA_THRESHOLD = 50  # pixels with alpha below this (0–255) are treated as background

COLOUR_MASK = "colour-mask"  # model_name for the Ultra-speed border-colour mask (no AI model)
_HYBRID_MAX_BLUR = 70        # hybrid: patches blurrier than this use the colour mask...
_HYBRID_MAX_SIDE = 150       # ...and so do patches with no side longer than this (px)
_METHOD_KEY = "mothbot_mask"  # PNG text chunk naming the method that made a _nobg.png

_rembg_session = None
_rembg_model_name: str | None = None


def _u2net_home() -> Path:
    """Directory where rembg looks for / caches its model weights."""
    return Path(os.path.expanduser(
        os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net"))
    ))


def _ensure_bundled_model(model_name: str) -> None:
    """Copy a model bundled with the app into the rembg cache so it isn't re-downloaded.

    Packaged builds ship the default model under ``<bundle>/models/`` (see the
    PyInstaller spec). rembg only looks in ``u2net_home()``, so on first run we copy
    the bundled file there; rembg's checksum check then passes and it skips the
    network download entirely. No-op when the model is already cached or when no
    bundled copy exists (dev runs fall back to rembg's normal download).
    """
    dest = _u2net_home() / f"{model_name}.onnx"
    if dest.exists():
        return

    # PyInstaller unpacks bundled datas under sys._MEIPASS; assets/ is the dev fallback.
    search_dirs = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        search_dirs.append(Path(meipass) / "models")
    search_dirs.append(Path(__file__).resolve().parents[1] / "assets")

    src = next((d / f"{model_name}.onnx" for d in search_dirs
                if (d / f"{model_name}.onnx").is_file()), None)
    if src is None:
        return

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Installing bundled {model_name} model into {dest.parent} (first run)...")
        shutil.copy2(src, dest)
    except Exception as e:
        print(f"  ⚠️ Could not install bundled model ({e}); rembg will download it instead.")


def _get_session(model_name: str = "birefnet-general-lite"):
    global _rembg_session, _rembg_model_name
    if _rembg_session is not None and _rembg_model_name == model_name:
        return _rembg_session

    _ensure_bundled_model(model_name)
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
    elif "OpenVINOExecutionProvider" in available:
        # "AUTO:GPU,CPU" lets the OpenVINO EP itself fall back to CPU if no
        # Intel GPU/NPU is present, instead of failing at inference time.
        providers = [
            ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"}),
            "CPUExecutionProvider",
        ]
        print("  ⚡ OpenVINO acceleration active (Intel GPU/XPU)")
    else:
        providers = ["CPUExecutionProvider"]

    # rembg sizes its thread pool from OMP_NUM_THREADS, which ultralytics sets
    # to 1 on import. Hide it so ONNX Runtime uses its own default (all
    # physical cores): ~1.3x faster on CPU, identical masks.
    omp_threads = os.environ.pop("OMP_NUM_THREADS", None)
    try:
        try:
            _rembg_session = new_session(model_name, providers=providers)
        except TypeError:
            _rembg_session = new_session(model_name)
    finally:
        if omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = omp_threads

    _rembg_model_name = model_name
    print(f"  {model_name} model ready.")
    return _rembg_session


def _is_identified(shape: dict) -> bool:
    """True if a bot or a human has identified this detection."""
    return bool(shape.get("identifier_bot") or shape.get("identifier_human"))


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


def _colour_mask_background(patch_path: str) -> Image.Image:
    bgr = cv2.imread(patch_path)
    if bgr is None:
        raise ValueError("could not read patch image")
    alpha = np.where(insect_mask(bgr), 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha]), "RGBA")


def _save_nobg(rgba: Image.Image, nobg_path: str, method: str) -> None:
    info = PngInfo()
    info.add_text(_METHOD_KEY, method)
    rgba.save(nobg_path, pnginfo=info)


def _nobg_method(nobg_path: str) -> str | None:
    """Method that made a _nobg.png; None for files made before it was recorded (always a model)."""
    with Image.open(nobg_path) as img:
        return img.info.get(_METHOD_KEY)


def _is_colour_mask(method: str | None) -> bool:
    return bool(method) and method.startswith("border-colour-mask/")


def _choose_method(patch_path: str, blur_score, model_name: str, hybrid: bool) -> str:
    """The colour mask for small or blurry patches when hybrid is on, else *model_name*."""
    if model_name == COLOUR_MASK or not hybrid:
        return model_name
    if isinstance(blur_score, (int, float)) and blur_score > _HYBRID_MAX_BLUR:
        return COLOUR_MASK
    with Image.open(patch_path) as img:
        if max(img.size) <= _HYBRID_MAX_SIDE:
            return COLOUR_MASK
    return model_name


def _needs_nobg(nobg_path: str, method: str, overwrite: bool) -> bool:
    if overwrite or not os.path.isfile(nobg_path):
        return True
    # A quick colour-mask result is redone when the model is now asked for; a
    # model result is never swapped for the rougher colour mask.
    return method != COLOUR_MASK and _is_colour_mask(_nobg_method(nobg_path))


def _format_rate(seconds: float) -> str:
    return f"{seconds:.2f} s/patch" if seconds >= 0.1 else f"{1000 * seconds:.0f} ms/patch"


def _make_nobgs(patches: list[str], method: str) -> tuple[int, int]:
    """Write a _nobg.png for each patch with *method*; prints progress every ~2 s. Returns (done, errors)."""
    if method == COLOUR_MASK:
        make, method_id = _colour_mask_background, COLOUR_MASK_METHOD
    else:
        make, method_id = (lambda p: _remove_background(p, model_name=method)), method

    total = len(patches)
    done = errors = 0
    t_start = time.monotonic()
    last_report = 0.0  # progress is printed every ~2 s: one line per patch flooded the UI log
    for patch_abs in patches:
        nobg = _nobg_path(patch_abs)
        try:
            _save_nobg(make(patch_abs), nobg, method_id)
            done += 1
        except Exception as e:
            errors += 1
            print(f"  ❌ [{done + errors}/{total}] {os.path.basename(patch_abs)}: {e}")
            continue
        now = time.monotonic()
        if now - last_report >= 2.0 or done + errors == total:
            last_report = now
            emit_preview(nobg)  # one per progress line: the UI takes one preview per log update
            remaining = total - done - errors
            avg = (now - t_start) / done
            eta_str = _format_eta(avg * remaining) if remaining else "done"
            print(f"  ✓ [{done}/{total}] {os.path.basename(patch_abs)} — {_format_rate(avg)} — ETA {eta_str}")
    return done, errors


def _count_foreground_pixels(nobg_path: str) -> tuple[int, str | None]:
    """Foreground pixel count and the method that made the _nobg.png."""
    with Image.open(nobg_path) as img:
        method = img.info.get(_METHOD_KEY)
        arr = np.asarray(img.convert("RGBA"))
    return int(np.sum(arr[:, :, 3] >= _ALPHA_THRESHOLD)), method


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
    only_identified: bool = False,
    hybrid: bool = True,
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
        If False, skip shapes that already have pixel_mass_pixels in the JSON
        (unless their _nobg.png was remade in this run).
    model_name:
        rembg model for background removal, or COLOUR_MASK for the quick
        border-colour mask (no AI model) on every patch.
    only_identified:
        If True, only measure patches that have an identification (bot or human),
        e.g. to skip patches ID left unidentified because they were too blurry.
    hybrid:
        If True, patches blurrier than 70 or with no side longer than 150 px use
        the colour mask; the rest use *model_name*.
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

    use_hybrid = hybrid and model_name != COLOUR_MASK
    if use_hybrid:  # the hybrid rule needs current blur scores (older datasets lack them)
        fill_missing_blur_scores(bot_pairs, _dataset_root, label="bot patches")

    # ── Load all JSONs and collect patch paths ────────────────────────────────
    # json_store: json_path → loaded dict (mutated in Phase 2, written at end)
    json_store = {}
    # all_patches: deduplicated ordered list of patch_abs paths that exist on disk
    all_patches = []
    _seen_patches: set[str] = set()
    patch_blur: dict[str, float | None] = {}

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
            if only_identified and not _is_identified(shape):
                continue
            patch_rel = shape.get("patch_path", "")
            if not patch_rel:
                continue
            patch_abs = resolve_patch_path(patch_rel, image_path, _dataset_root)
            if not os.path.isfile(patch_abs):
                continue
            if patch_abs not in _seen_patches:
                _seen_patches.add(patch_abs)
                all_patches.append(patch_abs)
                patch_blur[patch_abs] = shape.get("blur_score")

    if not all_patches:
        print("No patches found — nothing to process.")
        return

    # ── Phase 1: Background Removal ───────────────────────────────────────────
    clear_preview()
    methods = {p: _choose_method(p, patch_blur.get(p), model_name, use_hybrid) for p in all_patches}
    to_process = [p for p in all_patches if _needs_nobg(_nobg_path(p), methods[p], overwrite_nobg)]
    quick = [p for p in to_process if methods[p] == COLOUR_MASK]
    with_model = [p for p in to_process if methods[p] != COLOUR_MASK]
    total_bg = len(to_process)
    skipped_bg = len(all_patches) - total_bg

    if skipped_bg:
        print(f"\n── Phase 1: Background removal — {total_bg} patches ({skipped_bg} already have _nobg.png, skipping)")
    else:
        print(f"\n── Phase 1: Background removal — {total_bg} patches")
    if use_hybrid:
        print(f"   Hybrid optimization: {len(quick)} small or blurry patches → colour mask, "
              f"{len(with_model)} → {model_name}")

    bg_done = 0
    bg_errors = 0
    remade: set[str] = set()  # _nobg.png files written this run; Phase 2 recounts them
    for method, patches in ((COLOUR_MASK, quick), (model_name, with_model)):
        if not patches:
            continue
        if method == COLOUR_MASK:
            print(f"\n  Colour mask (no AI model): {len(patches)} patches")
        else:
            _get_session(method)  # load model upfront so ETA reflects only inference time
            print(f"\n  {method}: {len(patches)} patches")
        done, errors = _make_nobgs(patches, method)
        bg_done += done
        bg_errors += errors
        remade.update(_nobg_path(p) for p in patches)

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
            if only_identified and not _is_identified(shape):
                continue

            patch_rel = shape.get("patch_path", "")
            if not patch_rel:
                continue
            patch_abs = resolve_patch_path(patch_rel, image_path, _dataset_root)
            nobg = _nobg_path(patch_abs)

            if not overwrite_pixmass and "pixel_mass_pixels" in shape and nobg not in remade:
                px_skipped += 1
                continue
            if not os.path.isfile(nobg):
                continue

            try:
                px_count, method = _count_foreground_pixels(nobg)
                shape["pixel_mass_pixels"] = px_count
                if method:
                    shape["pixel_mass_method"] = method
                else:
                    shape.pop("pixel_mass_method", None)
                shape["pixel_mass_mm2"] = (
                    round(px_count / (pixels_per_mm ** 2), 4) if pixels_per_mm else None
                )
                shape["timestamp_pixel_mass"] = current_timestamp()
                changed = True
                px_done += 1
                if px_done % 500 == 0:
                    print(f"  ✓ {px_done} patches counted so far…")
            except Exception as e:
                px_errors += 1
                print(f"  ❌ {os.path.basename(patch_abs)}: {e}")

        if changed:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

    print(f"\n✅ Pixel Mass complete")
    print(f"   Phase 1 (bg removal): {bg_done} done, {skipped_bg} skipped, {bg_errors} errors")
    print(f"   Phase 2 (px count):   {px_done} done, {px_skipped} skipped, {px_errors} errors")
