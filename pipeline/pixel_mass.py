#!/usr/bin/env python3
"""
pixel_mass.py — Background removal and pixel-mass measurement for patches.

Detect already gives every patch a quick colour-mask _nobg.png and its pixel count
(``quick_nobg``). This module then:
  * applies a calibration (``apply_calibration``): colour-masks any patch that has
    no _nobg.png yet (older datasets) and sets every detection's area in mm²;
  * refines (``run``): re-does larger, sharper patches with an AI model.

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

COLOUR_MASK = "colour-mask"  # model_name for the quick border-colour mask (no AI model)

# BiRefNet-lite re-exported with a dynamic input size and ONNX Runtime's native
# DeformConv op (huggingface.co/senty-au/BiRefNet_lite-ONNX-dynamic; MIT, weights
# ZhengPeng7/BiRefNet_lite). At 1024 it matches rembg's fixed-1024 lite export
# (median mask IoU 0.99 on our patches) about 2x faster; 512 is ~5x faster again.
_DYNAMIC_LITE_NAME = "birefnet-lite-dynamic"
_DYNAMIC_LITE_URL = "https://huggingface.co/senty-au/BiRefNet_lite-ONNX-dynamic/resolve/main/onnx/model.onnx"
_DYNAMIC_LITE_SHA256 = "1e0da42f0fde010e32e938bad388457ecefe35806fde9d923421997861ae9391"
# Model choices that run on it, and the square input size each uses.
DYNAMIC_LITE_SIZES = {"birefnet-general-lite": 1024, "birefnet-lite-512": 512}
DEFAULT_MODEL = "birefnet-lite-512"
# "Fast, Split Model Approach": the lite model at 512 px, with a patch redone at
# 1024 px when the 512 px result
#   * has less than half the pixels the colour mask found (it lost most of the
#     insect — often one against a dark background), or
#   * is mostly semi-transparent (a faint haze rather than an outline).
# On 180 large, sharp bowedBarbo patches this redid 21% of them and caught 27 of
# the 35 where 512 and 1024 px disagreed badly; it averages ~1 s/patch vs ~2 s
# for 1024 px everywhere.
_FALLBACK_SHARE = 0.5
_FALLBACK_MAX_SEMI = 0.8   # share of the found pixels that are semi-transparent
_SEMI_ALPHA = 235          # alpha below this (and >= _ALPHA_THRESHOLD) counts as semi-transparent
_FALLBACK_MODEL = "birefnet-general-lite"  # the lite model at 1024 px
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_HYBRID_MAX_BLUR = 70        # hybrid: patches blurrier than this use the colour mask...
_HYBRID_MAX_SIDE = 150       # ...and so do patches with no side longer than this (px)
_METHOD_KEY = "mothbot_mask"  # PNG text chunk naming the method that made a _nobg.png

_rembg_session = None
_rembg_model_name: str | None = None
_dynamic_session = None


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


def _providers() -> list:
    """ONNX Runtime providers: a GPU when there is one, else the CPU."""
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
    return providers


def _ensure_dynamic_lite() -> Path:
    """Path to the dynamic BiRefNet-lite model: bundled with the app, cached, or
    downloaded once (181 MB) and checked against its published checksum."""
    _ensure_bundled_model(_DYNAMIC_LITE_NAME)
    dest = _u2net_home() / f"{_DYNAMIC_LITE_NAME}.onnx"
    if dest.is_file():
        return dest

    import hashlib
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".download")
    print("  Downloading the BiRefNet-lite model (181 MB, first use only)...")
    reported = [-1]

    def report(blocks, block_size, total):
        pct = int(100 * blocks * block_size / total) if total > 0 else 0
        if pct // 10 > reported[0] and pct < 100:
            reported[0] = pct // 10
            print(f"    {pct}%")

    urllib.request.urlretrieve(_DYNAMIC_LITE_URL, partial, reporthook=report)
    digest = hashlib.sha256()
    with open(partial, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != _DYNAMIC_LITE_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("The downloaded BiRefNet-lite model failed its checksum — please try again.")
    partial.replace(dest)
    return dest


def _get_dynamic_session():
    global _dynamic_session
    if _dynamic_session is None:
        import onnxruntime as _ort

        path = _ensure_dynamic_lite()
        print("Loading BiRefNet-lite background-removal model...")
        # Own SessionOptions: ONNX Runtime's default threads (all physical cores),
        # whatever OMP_NUM_THREADS ultralytics set.
        _dynamic_session = _ort.InferenceSession(str(path), _ort.SessionOptions(), providers=_providers())
        print("  BiRefNet-lite model ready.")
    return _dynamic_session


def _get_session(model_name: str = DEFAULT_MODEL):
    global _rembg_session, _rembg_model_name
    if model_name in DYNAMIC_LITE_SIZES:
        return _get_dynamic_session()
    if _rembg_session is not None and _rembg_model_name == model_name:
        return _rembg_session

    _ensure_bundled_model(model_name)
    print(f"Loading {model_name} background-removal model (first use may download weights)...")
    from rembg import new_session

    providers = _providers()

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


def _dynamic_lite_cutout(img: Image.Image, size: int) -> Image.Image:
    """Dynamic BiRefNet-lite at size x size, with rembg's exact BiRefNet pre- and
    post-processing and cutout, so results are comparable to its lite export."""
    from rembg.bg import naive_cutout

    sess = _get_dynamic_session()
    arr = np.array(img.resize((size, size), Image.Resampling.LANCZOS))
    arr = arr / max(np.max(arr), 1e-6)
    x = ((arr - np.array(_MEAN)) / np.array(_STD)).transpose(2, 0, 1)[None].astype(np.float32)
    pred = 1 / (1 + np.exp(-sess.run(None, {sess.get_inputs()[0].name: x})[0][:, 0, :, :]))
    lo, hi = np.min(pred), np.max(pred)
    pred = np.squeeze((pred - lo) / max(hi - lo, 1e-12))
    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L").resize(img.size, Image.Resampling.LANCZOS)
    return naive_cutout(img, mask)


def _model_cutout(patch_path: str, model_name: str = DEFAULT_MODEL) -> tuple[Image.Image, str]:
    """Background removal with *model_name*, and the method to record for it.
    Below 1024 px, a patch the model mostly misses is redone at 1024 px."""
    with Image.open(patch_path) as img:
        img_rgb = img.convert("RGB")
    size = DYNAMIC_LITE_SIZES.get(model_name)
    if size is None:
        from rembg import remove as rembg_remove
        return rembg_remove(img_rgb, session=_get_session(model_name)), _method_id(model_name)
    rgba = _dynamic_lite_cutout(img_rgb, size)
    if size < DYNAMIC_LITE_SIZES[_FALLBACK_MODEL] and _missed_insect(rgba, img_rgb):
        return _dynamic_lite_cutout(img_rgb, DYNAMIC_LITE_SIZES[_FALLBACK_MODEL]), _method_id(_FALLBACK_MODEL)
    return rgba, _method_id(model_name)


def _missed_insect(rgba: Image.Image, img_rgb: Image.Image) -> bool:
    """True when a low-resolution model result lost most of the insect or is mostly
    a semi-transparent haze (see _FALLBACK_SHARE / _FALLBACK_MAX_SEMI)."""
    alpha = np.asarray(rgba)[:, :, 3]
    found = np.count_nonzero(alpha >= _ALPHA_THRESHOLD)
    if found and np.count_nonzero((alpha >= _ALPHA_THRESHOLD) & (alpha < _SEMI_ALPHA)) > _FALLBACK_MAX_SEMI * found:
        return True
    expected = np.count_nonzero(insect_mask(cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)))
    return bool(expected) and found < _FALLBACK_SHARE * expected


def _method_id(model_name: str) -> str:
    """What a _nobg.png records as the method that made it."""
    if model_name == COLOUR_MASK:
        return COLOUR_MASK_METHOD
    if model_name in DYNAMIC_LITE_SIZES:
        return f"{_DYNAMIC_LITE_NAME}@{DYNAMIC_LITE_SIZES[model_name]}"
    return model_name


def _colour_mask_rgba(bgr: np.ndarray) -> Image.Image:
    alpha = np.where(insect_mask(bgr), 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha]), "RGBA")


def _colour_mask_background(patch_path: str) -> Image.Image:
    bgr = cv2.imread(patch_path)
    if bgr is None:
        raise ValueError("could not read patch image")
    return _colour_mask_rgba(bgr)


def _area_mm2(pixels: int, pixels_per_mm: float | None) -> float | None:
    return round(pixels / (pixels_per_mm ** 2), 4) if pixels_per_mm else None


def quick_nobg(patch_bgr: np.ndarray, patch_path: str, pixels_per_mm: float | None = None) -> dict:
    """Colour-mask a patch that is already in memory, save its _nobg.png and return
    the pixel-mass fields for its detection shape. Detect calls this for every patch
    (a few ms each); the area in mm² is None until the collection is calibrated."""
    rgba = _colour_mask_rgba(patch_bgr)
    # Fast compression: ~3x quicker to write than the default, files ~1.7x bigger.
    _save_nobg(rgba, _nobg_path(patch_path), COLOUR_MASK_METHOD, compress_level=1)
    pixels = int(np.count_nonzero(np.asarray(rgba)[:, :, 3]))
    return {
        "pixel_mass_pixels": pixels,
        "pixel_mass_mm2": _area_mm2(pixels, pixels_per_mm),
        "pixel_mass_method": COLOUR_MASK_METHOD,
        "timestamp_pixel_mass": current_timestamp(),
    }


def _save_nobg(rgba: Image.Image, nobg_path: str, method: str, compress_level: int = 6) -> None:
    info = PngInfo()
    info.add_text(_METHOD_KEY, method)
    rgba.save(nobg_path, pnginfo=info, compress_level=compress_level)


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
        make = lambda p: (_colour_mask_background(p), COLOUR_MASK_METHOD)  # noqa: E731
    else:
        make = lambda p: _model_cutout(p, method)  # noqa: E731
    primary = _method_id(method)

    total = len(patches)
    done = errors = fallbacks = 0
    t_start = time.monotonic()
    last_report = 0.0  # progress is printed every ~2 s: one line per patch flooded the UI log
    for patch_abs in patches:
        nobg = _nobg_path(patch_abs)
        try:
            rgba, method_id = make(patch_abs)
            _save_nobg(rgba, nobg, method_id)
            done += 1
            fallbacks += method_id != primary
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
    if fallbacks:
        print(f"  ↻ {fallbacks} patch(es) redone at 1024 px: the 512 px result missed most of the insect or was mostly see-through")
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
    model_name: str = DEFAULT_MODEL,
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
    px_area = 0  # kept counts whose area changed with the calibration

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
                # Count kept, but the area follows the current calibration.
                area = _area_mm2(shape["pixel_mass_pixels"], pixels_per_mm)
                if pixels_per_mm and shape.get("pixel_mass_mm2") != area:
                    shape["pixel_mass_mm2"] = area
                    changed = True
                    px_area += 1
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
                shape["pixel_mass_mm2"] = _area_mm2(px_count, pixels_per_mm)
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
    print(f"   Phase 2 (px count):   {px_done} counted, {px_skipped} kept, {px_errors} errors")
    if px_area:
        print(f"   Areas (mm²) updated from the calibration for {px_area} kept counts")


def apply_calibration(input_path: str, dataset_root: str | None, pixels_per_mm: float) -> None:
    """Give every detection in *input_path* its real-world area from a calibration.

    Detect already colour-masks every patch and records its pixel count; patches
    that predate that (older datasets) get the quick colour mask first. Existing
    _nobg.png files and counts are kept (including ones refined with a model), and
    pixel_mass_mm2 is set from each count. The calibration is saved for the
    collection if it has none, so later refinements use it too.
    """
    processed_root = get_processed_folder(input_path, dataset_root or input_path)
    calib = load_calibration(processed_root) or {}
    if calib.get("pixels_per_mm") != pixels_per_mm:
        save_calibration(processed_root, {**calib, "pixels_per_mm": pixels_per_mm,
                                          "calibration_date": current_timestamp()})
    run(
        input_path=input_path,
        dataset_root=dataset_root,
        pixels_per_mm=pixels_per_mm,
        overwrite_nobg=False,
        overwrite_pixmass=False,
        model_name=COLOUR_MASK,
        hybrid=False,
    )
