"""
Blurriness score for detection patches.

Some researchers count every detection that could be an insect, blurry smudges
included; others only want patches they can identify. A per-patch blurriness
score lets both work from one dataset: ID can skip patches blurrier than a
threshold, and Mothbot Classify can sort or filter by it.

Metric: the perceptual blur measure of Crété-Roffet et al. (2007), "The blur
effect: perception and estimation with a new no-reference perceptual blur
metric". Re-blur the patch with a 1-D box filter and measure how much
neighbour-to-neighbour variation it *loses*: a sharp image loses a lot, an
already-blurry one barely changes. It is contrast-invariant (a ratio), so a
faint smudge and a high-contrast beetle are judged on edge sharpness rather
than brightness.

Measured over the insect only. On the whole patch, the background dominates:
paper-fibre texture and sensor noise are genuinely sharp, so motion streaks
and smudges (mostly background) scored as the sharpest patches of all. The
insect is found as the pixels whose smoothed brightness differs from the
patch border's (tight crops can be mostly insect, so not the whole-patch
median), grown a few pixels to take in its edges.

Checked by eye on the 20 sharpest, 20 blurriest and an evenly spaced spread
of 3,000 real patches. Known limits: whole-sheet false detections score sharp
(their paper texture really is in focus), and a few faint streaks, mostly on
the dark background off the sheet, still pass as sharp. ~3 ms per patch.

The raw measure (0 sharp .. 1 blurry) spans ~0.32-0.68 on Mothbox patches
(1st-99th percentile), so it is linearly rescaled to a 0-100 blurriness score
with fixed bounds. The bounds are constants — not per-dataset — so scores are
comparable across datasets.
"""

import json
import os

import cv2
import numpy as np

BLUR_METHOD = "crete-roffet-2007/object-v2"

# Raw-measure bounds mapped to 0 and 100. Fixed so scores mean the same thing in
# every dataset; chosen from the 1st/99th percentiles of real patches with margin.
_RAW_SHARPEST = 0.30
_RAW_BLURRIEST = 0.75

# Size of the re-blur box filter (the reference implementation's default).
_H_SIZE = 11

# How far the insect mask is grown to take in its (possibly soft) edges.
_GROW_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def _object_region(gray):
    """Mask of the insect: pixels whose smoothed brightness differs from the border's."""
    h, w = gray.shape
    ring = np.concatenate([gray[:3].ravel(), gray[-3:].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()])
    diff = np.abs(cv2.GaussianBlur(gray, (0, 0), 2.0) - float(np.median(ring)))
    mask = diff > max(0.04, 0.3 * float(np.percentile(diff, 99)))
    mask = cv2.dilate(mask.astype(np.uint8), _GROW_KERNEL).astype(bool)
    inner = np.zeros_like(mask)
    inner[2 : h - 1, 2 : w - 1] = True  # Sobel/box-filter border effects stay out
    region = mask & inner
    # Nothing stands out from the background: judge the whole patch instead.
    return region if region.sum() >= max(25, 0.01 * h * w) else inner


def raw_blur_effect(patch_bgr):
    """Crété-Roffet blur measure of a BGR (or grayscale) patch's insect: 0 sharp .. 1 blurry."""
    if patch_bgr is None or patch_bgr.size == 0:
        return 1.0
    gray = patch_bgr if patch_bgr.ndim == 2 else cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape
    if h < 5 or w < 5:
        return 1.0

    region = _object_region(gray)
    scores = []
    for axis in (0, 1):
        # cv2 kernel sizes are (width, height): blur ALONG the derivative's axis.
        ksize = (1, _H_SIZE) if axis == 0 else (_H_SIZE, 1)
        blurred = cv2.blur(gray, ksize, borderType=cv2.BORDER_REFLECT)
        dx, dy = (0, 1) if axis == 0 else (1, 0)
        sharp = np.abs(cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=3))
        soft = np.abs(cv2.Sobel(blurred, cv2.CV_32F, dx, dy, ksize=3))
        total = float(sharp[region].sum())
        if total <= 0:
            scores.append(1.0)  # no edges at all — nothing to see
            continue
        lost = float(np.maximum(0.0, sharp - soft)[region].sum())
        scores.append(abs(total - lost) / total)
    return max(scores)


def blur_score(patch_bgr):
    """Blurriness 0 (sharpest) .. 100 (blurriest), rounded to 0.1."""
    raw = raw_blur_effect(patch_bgr)
    scaled = (raw - _RAW_SHARPEST) / (_RAW_BLURRIEST - _RAW_SHARPEST)
    return round(float(np.clip(scaled, 0.0, 1.0)) * 100.0, 1)


def set_blur_on_shape(shape, patch_bgr):
    """Record the patch's blurriness on its detection shape (in place)."""
    shape["blur_score"] = blur_score(patch_bgr)
    shape["blur_method"] = BLUR_METHOD


def shape_needs_blur(shape):
    return not isinstance(shape.get("blur_score"), (int, float)) or shape.get("blur_method") != BLUR_METHOD


def _with_archived_runs(pairs):
    """Yield each (image, json) pair plus earlier detection runs archived beside it.

    Re-running Detect with a new model archives the previous run's JSON as
    ``<photo>_botdetection_<model>.json`` next to the current one; its patches are
    still shown in Classify (as another detection run), so score them too.
    """
    import glob

    for image_path, json_path in pairs:
        yield image_path, json_path
        if json_path.endswith("_botdetection.json"):
            stem = glob.escape(json_path[: -len("_botdetection.json")])
            for archived in sorted(glob.glob(f"{stem}_botdetection_*.json")):
                yield image_path, archived


def fill_missing_blur_scores(bot_pairs, dataset_root, label="patches"):
    """Score every patch whose detection shape has no (current-method) blur score.

    For datasets detected before blur scoring existed. Reads each patch image from
    disk once and writes scores back into its detection JSON, so this is a one-time
    cost (~3 ms/patch). Returns (scored, missing_patch_files).
    """
    from core.paths import resolve_patch_path

    todo = []
    for image_path, json_path in _with_archived_runs(bot_pairs):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            continue
        if any(shape_needs_blur(s) for s in data.get("shapes", []) if s.get("patch_path")):
            todo.append((image_path, json_path))
    if not todo:
        return 0, 0

    print(f"  Scoring blurriness for {label} in {len(todo)} detection file(s) (one-time for older datasets)...")
    scored = missing = 0
    for index, (image_path, json_path) in enumerate(todo, start=1):
        with open(json_path) as f:
            data = json.load(f)
        changed = False
        for shape in data.get("shapes", []):
            patch_rel = shape.get("patch_path")
            if not patch_rel or not shape_needs_blur(shape):
                continue
            patch_path = resolve_patch_path(patch_rel, image_path, dataset_root)
            patch = cv2.imread(patch_path) if os.path.isfile(patch_path) else None
            if patch is None:
                missing += 1
                continue
            set_blur_on_shape(shape, patch)
            scored += 1
            changed = True
        if changed:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        if index % 100 == 0 or index == len(todo):
            print(f"    🔎 blurriness: {index}/{len(todo)} files ({scored} patches scored)")
    return scored, missing
