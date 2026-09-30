"""
Blurriness score for detection patches.

Some researchers count every detection that could be an insect, blurry smudges
included; others only want patches they can identify. A per-patch blurriness
score lets both work from one dataset: ID can skip patches blurrier than a
threshold, and Mothbot Classify can sort or filter by it.

A patch is hard to identify when its blur is large *relative to the insect*:
an out-of-focus insect, an insect so small that the camera's ordinary blur
swallows its detail, or a motion streak. Two measurements, both read straight
from the image's derivatives in native pixels, cover these. Nothing is fitted
to data and nothing depends on the background's colour or brightness (the
border colour is only used to find the insect).

1. Edge blur relative to insect size (defocus, too small). At the insect's
   strongest edges, compare the gradient at two Gaussian scales (1 and 2 px).
   For an edge blurred by sigma_b, g(s) is proportional to 1/sqrt(s^2 + sigma_b^2),
   so the ratio q = g(1)/g(2) gives sigma_b^2 = (4 - q^2) / (q^2 - 1), whatever
   the edge's contrast (Elder & Zucker 1998; Zhuo & Sim 2011). The median over
   the edges, as a 10-90% edge width (2.56 sigma), divided by the insect's size
   (square root of its area), is the blurriness: "the blur is N% of the insect".

2. Motion streak. Motion smears the insect along its direction of travel, so
   the image (smooth or jaggedly rolling-shutter streaked) keeps matching
   itself when shifted along that direction, but not across it. From the
   autocorrelation of the insect's derivative field, the shift at which the
   match drops to half is found in 12 directions; the ratio of the longest to
   the shortest is `motion_streak`. Above STREAK_RATIO (6) the patch is a motion
   streak and scores 100: whatever its edges, a smear is not identifiable.

Checked against 200 patches a person rated sharp / usable / too blurry, from
10 datasets (evaluation only; nothing was fitted to them). The true camera blur
at edges was ~2 px whether a patch was rated sharp or too blurry: blurriness
was mostly about that blur relative to the insect's size, as scored here.
Blurriness medians: sharp 5, usable 13, too blurry 19; AUC 0.84 for too
blurry vs the rest (0.75-0.96 per dataset). A threshold of 15 skipped 69% of
too-blurry patches, 27% of usable and none of the sharp ones; 20 skipped
48% / 6% / 0%. The streak rule caught 4 of 9 motion patches (of the rest, three
are small blurry blobs, which the blur part scores, and two are whole-frame
detections) and flagged 1 of 130 non-blurry patches (a thin, elongated insect).
Also checked on patches tagged ERROR_Blur in Classify (843, two datasets) vs
patches identified there (181): AUC 0.92; the streak rule fired on half the
ERROR_Blur patches and on 1 identified one. Re-check with
tools/blur_calibration/evaluate.py. Known limits: whole-sheet false detections
of empty paper score as sharp; a large insect with moderate defocus (edges ~2x
the camera's usual blur) scores low, because its blur is small relative to its size.

~0.5 ms per typical patch; tens of ms for the rare patches over ~300 px.
"""

import json
import os

import cv2
import numpy as np

BLUR_METHOD = "edge-blur-relative+streak/v4"

STREAK_RATIO = 6.0  # self-similar this many times further along one direction than across: a motion streak
_EDGE_WIDTH_PER_SIGMA = 2.56  # 10-90% rise distance of a Gaussian-blurred step, in sigmas
_GROW_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_STREAK_ANGLES = 12
_STREAK_MAX_SHIFT = 400
_SCALES = (1.0, 2.0, 4.0, 8.0)  # Gaussian scales (px) for reading edge blur


def _colour_difference(patch_bgr, valid):
    """Lab colour distance of each pixel from the paper colour (median of the patch's 3-px border)."""
    lab = cv2.cvtColor(patch_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    border = np.zeros(valid.shape, bool)
    border[:3] = border[-3:] = True
    border[:, :3] = border[:, -3:] = True
    ring = lab[border & valid] if (border & valid).any() else lab[border]
    return np.linalg.norm(lab - np.median(ring, axis=0), axis=2)


def _valid_pixels(patch_bgr):
    """Pixels that are photo, not the black fill Detect puts where a box runs off the image
    (that fill's straight, sharp edge would otherwise read as a sharp edge or a streak)."""
    photo = patch_bgr.max(axis=2) > 0
    if photo.all():
        return photo, None
    distance = cv2.distanceTransform(photo.astype(np.uint8), cv2.DIST_L2, 3)
    return photo, distance


def _percentile(values, q):
    """np.percentile(values, q) (linear interpolation), via a partial sort: much cheaper on small arrays."""
    flat = values.ravel()
    position = (flat.size - 1) * q / 100.0
    lo = int(position)
    hi = min(lo + 1, flat.size - 1)
    part = np.partition(flat, (lo, hi))
    return float(part[lo] + (part[hi] - part[lo]) * (position - lo))


def _insect_mask(smooth2):
    """The insect: pixels whose colour difference (smoothed at 2 px) clearly stands out."""
    return smooth2 > max(4.0, 0.3 * _percentile(smooth2, 99))


def _sobel(smooth):
    return (cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0,
            cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0)


def _gradient(image, sigma):
    return _sobel(cv2.GaussianBlur(image, (0, 0), sigma))


def _edge_blur_percent(diff, smooth2, region, g1, size):
    """Median edge blur (10-90% width) at the insect's strongest edges, as % of the insect's size.

    Each edge's blur is read from the finest pair of scales (s, 2s) on the ladder
    1-2-4-8 px whose estimate falls within [s/2, 2*2s]: a pair of scales can only
    resolve blur comparable to them (Elder & Zucker's minimum reliable scale).
    """
    h, w = diff.shape
    inner = region.copy()
    inner[:3] = inner[-3:] = False
    inner[:, :3] = inner[:, -3:] = False
    if inner.sum() < 25:
        return 100.0
    mags = [g1, cv2.magnitude(*_sobel(smooth2))]  # 2 px scale = the mask's smoothing, reused
    edges = inner & (mags[1] >= _percentile(mags[1][inner], 90))
    sigma_b = np.full(int(edges.sum()), np.nan)
    for i in range(len(_SCALES) - 1):
        if not np.isnan(sigma_b).any():
            break  # every edge resolved: coarser scales not needed (most sharp patches)
        if len(mags) <= i + 1:
            mags.append(cv2.magnitude(*_gradient(diff, _SCALES[i + 1])))
        s1, s2 = _SCALES[i], _SCALES[i + 1]
        q2 = np.clip((mags[i][edges] / np.maximum(mags[i + 1][edges], 1e-9)) ** 2, 1.0 + 1e-4, None)
        estimate = np.sqrt(np.clip((s2 * s2 - q2 * s1 * s1) / (q2 - 1.0), 0.0, 1e4))
        resolved = np.isnan(sigma_b) & (estimate <= 2 * s2) & ((estimate >= s1 / 2) if i else True)
        sigma_b[resolved] = estimate[resolved]
    sigma_b[np.isnan(sigma_b)] = 2 * _SCALES[-1]  # blurrier than the ladder reaches
    return min(100.0, 100.0 * _EDGE_WIDTH_PER_SIGMA * float(np.median(sigma_b)) / size)


def _streak_ratio(region, gx, gy):
    """Longest / shortest shift (over 12 directions) before the derivative field stops matching itself."""
    ys, xs = np.nonzero(region)
    if len(xs) < 25:
        return 1.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1  # crop to the insect (no resizing)
    keep = region[y0:y1, x0:x1]
    field = np.zeros((y1 - y0, x1 - x0, 2), np.float32)  # gx + i*gy: one transform for both
    field[..., 0] = np.where(keep, gx[y0:y1, x0:x1], 0)
    field[..., 1] = np.where(keep, gy[y0:y1, x0:x1], 0)
    h, w = keep.shape
    H, W = cv2.getOptimalDFTSize(2 * h), cv2.getOptimalDFTSize(2 * w)
    padded = np.zeros((H, W, 2), np.float32)
    padded[:h, :w] = field
    spectrum = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    power = cv2.mulSpectrums(spectrum, spectrum, 0, conjB=True)
    corr = cv2.idft(power, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # real part = autocorr(gx) + autocorr(gy)
    corr /= max(float(corr[0, 0]), 1e-12)
    # Sample all 12 rays in one call; the correlation is periodic, so negative shifts wrap.
    shifts = np.arange(0, min(max(h, w), _STREAK_MAX_SHIFT) + 1, dtype=np.float32)
    angles = np.linspace(0, np.pi, _STREAK_ANGLES, endpoint=False, dtype=np.float32)
    px = (np.cos(angles)[:, None] * shifts[None]).astype(np.float32)
    py = (np.sin(angles)[:, None] * shifts[None]).astype(np.float32)
    profiles = cv2.remap(corr, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    below = profiles < 0.5
    lengths = np.where(below.any(axis=1), below.argmax(axis=1), len(shifts) - 1).astype(np.float64)
    return float(lengths.max() / max(lengths.min(), 0.5))


def blur_fields(patch_bgr):
    """{"blur_score": 0-100, "motion_streak": ratio, "blur_method": ...} for a BGR patch."""
    if patch_bgr is None or patch_bgr.size == 0 or min(patch_bgr.shape[:2]) < 8:
        return {"blur_score": 100.0, "motion_streak": 1.0, "blur_method": BLUR_METHOD}
    if patch_bgr.ndim == 2:
        patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
    photo, distance = _valid_pixels(patch_bgr)
    diff = _colour_difference(patch_bgr, photo)
    smooth2 = cv2.GaussianBlur(diff, (0, 0), 2.0)
    mask = _insect_mask(smooth2)
    if distance is not None:
        mask &= distance > 3
    if mask.sum() < 16:  # nothing stands out from the paper: nothing to identify
        return {"blur_score": 100.0, "motion_streak": 1.0, "blur_method": BLUR_METHOD}
    region = cv2.dilate(mask.astype(np.uint8), _GROW_KERNEL).astype(bool)
    if distance is not None:
        region &= distance > 3  # keep the black fill's edge out of the gradients
    region[:3] = region[-3:] = False
    region[:, :3] = region[:, -3:] = False
    gx, gy = _gradient(diff, 1.0)
    streak = _streak_ratio(region, gx, gy)
    if streak > STREAK_RATIO:
        score = 100.0
    else:
        score = _edge_blur_percent(diff, smooth2, region, cv2.magnitude(gx, gy), float(np.sqrt(mask.sum())))
    return {"blur_score": round(score, 1), "motion_streak": round(streak, 1), "blur_method": BLUR_METHOD}


def blur_score(patch_bgr):
    """Blurriness 0 (sharp) .. 100: edge blur as % of the insect's size; motion streaks score 100."""
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
    cost (~1 ms/patch including the read). Returns (scored, missing_patch_files).
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
