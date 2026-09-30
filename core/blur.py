"""
Blurriness score for detection patches.

Some researchers count every detection that could be an insect, blurry smudges
included; others only want patches they can identify. A per-patch blurriness
score lets both work from one dataset: ID can skip patches blurrier than a
threshold, and Mothbot Classify can sort or filter by it.

Measure: a directional variant of the classic "variance of the Laplacian".

1. Resize the patch to a fixed number of pixels (48 x 48 = 2,304), keeping
   its shape. Blur only matters relative to the insect's size (a tiny bug is
   as unidentifiable as a defocused moth), and at one pixel count a big sharp
   moth and a small smudge are comparable, whatever their aspect ratio.
   Without this step Laplacian variance barely beat chance on real patches
   (dividing it by the pixel count made it worse than chance). The exact
   target hardly matters: 32^2 to 96^2 scored within a few hundredths.
2. Take the second derivative (Laplacian-style fine detail) along 8
   directions and its mean energy in each.
   The energy ignores a 2-px margin at the crop border (where a streak or
   body is cut off, the crop edge reads as a hard edge) and drops the
   strongest 5% of pixels: one straight line that isn't the insect, such as
   the edge of the sheet, would otherwise supply most of the "detail" (in one
   streak it supplied half). A sharp insect's detail is spread over wings,
   legs and body, so trimming barely changes it.
3. Sharpness = the energy in the *weakest* direction. Defocus and small size
   remove fine detail in every direction; motion blur removes it along the
   direction of travel, however sharp the streak's lines are across it. So
   one number catches both, with no separate streak rule.

The raw value is shown on a fixed log scale: log10 energy 3.25 (sharpest
~1% of patches across 10 datasets) is 0 and 1.4 (blurriest ~1%) is 100. The
constants are the same for every dataset, so scores are comparable; ranking
is unaffected by the scale. `motion_streak` (sharpest-direction energy /
weakest-direction energy) is recorded too: high means the detail survives in
one direction only, as in a motion streak.

Checked (nothing fitted) against hand labels: 200 patches rated from 10
datasets, AUC 0.91 for "too blurry" (focus 0.91, too small 0.94, motion
0.80; 0.75-1.00 per dataset); bowedBarbo 2026-06-24, fully reviewed in
Classify, ERROR_Blur + Error:Motion vs untouched, AUC 0.94 (motion alone
0.91; catching 70% of the tagged patches also skips 3.1% of the untouched
ones). Re-check with tools/blur_calibration/evaluate.py. ~0.2 ms per
typical patch.
"""

import json
import os

import cv2
import numpy as np

BLUR_METHOD = "directional-laplacian-area48-trim/v7"

_STANDARD_PIXELS = 48 * 48  # every patch is compared at this many pixels, shape kept
_DIRECTIONS = np.linspace(0, np.pi, 8, endpoint=False)
_LOG_SHARPEST = 3.25  # log10 weakest-direction energy scored 0
_LOG_BLURRIEST = 1.4  # ... scored 100
_MARGIN = 2  # px at the standard size, left out at the crop border
_TRIM = 0.05  # strongest share of pixels left out of each direction's energy


def _standard_gray(patch_bgr):
    """Grayscale at the standard pixel count. Detect's black fill (where a box ran off the
    photo) is replaced by the patch's median colour first, so its artificial edge
    doesn't count as detail."""
    if patch_bgr.ndim == 2:
        patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
    fill = patch_bgr.max(axis=2) == 0
    if fill.any() and not fill.all():
        patch_bgr = patch_bgr.copy()
        patch_bgr[fill] = np.median(patch_bgr[~fill], axis=0)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    scale = float(np.sqrt(_STANDARD_PIXELS / (h * w)))
    size = (max(3, round(w * scale)), max(3, round(h * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def directional_detail(patch_bgr):
    """Mean second-derivative energy along each of 8 directions, at the standard pixel count."""
    gray = _standard_gray(patch_bgr)
    dxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    maps = []
    for angle in _DIRECTIONS:
        c, s = np.cos(angle), np.sin(angle)
        energy = (c * c * dxx + 2 * c * s * dxy + s * s * dyy) ** 2
        if min(energy.shape) > 2 * _MARGIN + 2:
            energy = energy[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN]
        maps.append(energy.ravel())
    maps = np.stack(maps)  # one row per direction
    keep = max(1, int(maps.shape[1] * (1 - _TRIM)))
    energies = np.partition(maps, keep - 1, axis=1)[:, :keep].mean(axis=1)
    return energies.astype(np.float64)


def blur_fields(patch_bgr):
    """{"blur_score": 0-100, "motion_streak": ratio, "blur_method": ...} for a BGR patch."""
    if patch_bgr is None or patch_bgr.size == 0 or min(patch_bgr.shape[:2]) < 3:
        return {"blur_score": 100.0, "motion_streak": 1.0, "blur_method": BLUR_METHOD}
    energies = directional_detail(patch_bgr)
    weakest = float(energies.min())
    scaled = (_LOG_SHARPEST - np.log10(weakest + 1e-6)) / (_LOG_SHARPEST - _LOG_BLURRIEST)
    return {
        "blur_score": round(float(np.clip(scaled, 0.0, 1.0)) * 100.0, 1),
        "motion_streak": round(float(energies.max() / max(weakest, 1e-6)), 1),
        "blur_method": BLUR_METHOD,
    }


def blur_score(patch_bgr):
    """Blurriness 0 (sharp) .. 100 (blurriest), on a fixed scale."""
    return blur_fields(patch_bgr)["blur_score"]


def set_blur_on_shape(shape, patch_bgr):
    """Record the patch's blurriness on its detection shape (in place)."""
    shape.update(blur_fields(patch_bgr))


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
    cost (dominated by reading the patch images). Returns (scored, missing_patch_files).
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
