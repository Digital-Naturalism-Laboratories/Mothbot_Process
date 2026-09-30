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
   The energy ignores a margin of 10% of each side at the crop border (where a
   streak or body is cut off, the crop edge reads as a hard edge) and drops the
   strongest 5% of pixels: one straight line that isn't the insect, such as
   the edge of the sheet, would otherwise supply most of the "detail" (in one
   streak it supplied half). A sharp insect's detail is spread over wings,
   legs and body, so trimming barely changes it.
3. Sharpness = the energy in the *weakest* direction. Defocus and small size
   remove fine detail in every direction; motion blur removes it along the
   direction of travel, however sharp the streak's lines are across it. So
   one number catches both, with no separate streak rule.

The raw value is shown on a fixed log scale: log10 energy 3.3 (sharpest
~1% of patches across 10 datasets) is 0 and 1.45 (blurriest ~1%) is 100. The
constants are the same for every dataset, so scores are comparable; ranking
is unaffected by the scale. This is `blur_homogeneous`.

4. Motion: the FFT "sunshine" of the whole patch at full resolution. A sharp
   insect's spectrum radiates in many directions (edges every way); a motion
   streak's collapses into one or two lines. Measured as the evenness of
   spectral power over 18 directions (normalised entropy, 1 = sunshine) in a
   band of mid frequencies. `blur_motion` = 150 x (1 - sunshine), capped at 100.
   (Untouched patches sit near 0.93, motion streaks near 0.6.)

`blur_score` ("ugly score") = max(blur_homogeneous, blur_motion): whichever
problem is worse; `blur_winner` says which one set it.

`blur_type` says what kind of blur a patch has, if it is blurry: "motion" when
its fine detail is at least 3x stronger in one direction than in another
(`blur_direction_ratio`, from step 3's energies), else "homogeneous" (out of
focus or too small). Motion wipes out detail along the direction of travel
only; defocus wipes it out everywhere. On bowedBarbo 2026-06-24 this labelled
95% of Error:Motion patches "motion" and 7% of ERROR_Blur ones (AUC 0.97).
It describes the kind of blur, not how bad it is: a sharp, elongated insect
can have a high ratio too.

Checked against hand labels (the 150 was the only choice made on them; 1.5x
to 2x scored the same): 200 patches rated from 10 datasets, AUC 0.93 for
"too blurry" (focus 0.91, motion 0.92); bowedBarbo 2026-06-24, fully
reviewed in Classify, ERROR_Blur + Error:Motion vs untouched, AUC 0.955
(motion alone 0.95); ERROR_Blur vs identified in KrkCreate, AUC 0.95.
Re-check with tools/blur_calibration/evaluate.py. ~0.3-1 ms per patch.
"""

import json
import os
from functools import lru_cache

import cv2
import numpy as np
import scipy.fft  # float32 FFT (numpy's is always float64); scipy ships with hdbscan/scikit-learn

BLUR_METHOD = "ugly-max-detail-fft/v8"

_STANDARD_PIXELS = 48 * 48  # every patch is compared at this many pixels, shape kept
_DIRECTIONS = np.linspace(0, np.pi, 8, endpoint=False)
_LOG_SHARPEST = 3.3  # log10 weakest-direction energy scored 0
_LOG_BLURRIEST = 1.45  # ... scored 100
_MARGIN = 0.10  # share of each side left out at the crop border (so long thin patches are treated evenly)
_TRIM = 0.05  # strongest share of pixels left out of each direction's energy
_FFT_BINS = 18  # spectrum directions (10 degrees each)
_FFT_BAND = (0.05, 0.35)  # spatial frequencies (cycles/px) whose directions count
_MOTION_SCALE = 150.0  # blur_motion = this x (1 - sunshine)
_MOTION_RATIO = 3.0  # blur_type = "motion" at or above this strongest/weakest detail ratio


def _full_gray(patch_bgr):
    """Grayscale at full size. Detect's black fill (where a box ran off the photo) is
    replaced by the patch's median colour first, so its artificial edge doesn't count."""
    if patch_bgr.ndim == 2:
        patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) < gray.size:  # has pure-black pixels: look for fill (rare, so checked only then)
        fill = patch_bgr.max(axis=2) == 0
        if fill.any() and not fill.all():
            patch_bgr = patch_bgr.copy()
            patch_bgr[fill] = np.median(patch_bgr[~fill], axis=0)
            gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _standard_gray(patch_bgr, gray=None):
    """Grayscale at the standard pixel count, shape kept."""
    gray = _full_gray(patch_bgr) if gray is None else gray
    h, w = gray.shape
    scale = float(np.sqrt(_STANDARD_PIXELS / (h * w)))
    size = (max(3, round(w * scale)), max(3, round(h * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def directional_detail(patch_bgr, gray=None):
    """Mean second-derivative energy along each of 8 directions, at the standard pixel count."""
    gray = _standard_gray(patch_bgr, gray)
    dxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    maps = []
    for angle in _DIRECTIONS:
        c, s = np.cos(angle), np.sin(angle)
        energy = (c * c * dxx + 2 * c * s * dxy + s * s * dyy) ** 2
        h, w = energy.shape
        my, mx = max(1, round(_MARGIN * h)), max(1, round(_MARGIN * w))
        if h > 2 * my + 2 and w > 2 * mx + 2:
            energy = energy[my : h - my, mx : w - mx]
        maps.append(energy.ravel())
    maps = np.stack(maps)  # one row per direction
    keep = max(1, int(maps.shape[1] * (1 - _TRIM)))
    energies = np.partition(maps, keep - 1, axis=1)[:, :keep].mean(axis=1)
    return energies.astype(np.float64)


@lru_cache(maxsize=512)
def _fft_bins(h, w):
    """For an h x w patch's half spectrum (rfft2): flat indices inside the frequency band, their
    direction bin, and a weight of 2 for columns that stand for a mirrored pair of frequencies
    (the real spectrum is symmetric, so directions 0-180 degrees need only half of it)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    shape = (h, fx.shape[1])
    radius = np.hypot(fx, fy)
    band = np.flatnonzero((radius >= _FFT_BAND[0]) & (radius <= _FFT_BAND[1]))
    angle = (np.degrees(np.arctan2(np.broadcast_to(fy, shape), np.broadcast_to(fx, shape))) % 180.0).ravel()[band]
    bins = np.minimum((angle / (180.0 / _FFT_BINS)).astype(np.int64), _FFT_BINS - 1)
    pair = np.full(shape[1], 2.0)
    pair[0] = 1.0
    if w % 2 == 0:
        pair[-1] = 1.0  # the Nyquist column has no mirror either
    weight = np.broadcast_to(pair[None, :], shape).ravel()[band]
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return band, bins, weight, window


def fft_sunshine(gray):
    """How evenly the spectrum's power spreads over directions: 1 = radiant 'sunshine' (edges
    every way, as on a sharp insect); low = one or two lines (a motion streak)."""
    h, w = gray.shape
    if h < 8 or w < 8:
        return 1.0
    band, bins, weight, window = _fft_bins(h, w)
    spectrum = scipy.fft.rfft2((gray - np.float32(gray.mean())) * window)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).ravel()
    hist = np.bincount(bins, weights=power[band] * weight, minlength=_FFT_BINS)
    total = hist.sum()
    if total <= 0:
        return 1.0
    p = hist[hist > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(_FFT_BINS))


def blur_fields(patch_bgr):
    """{"blur_score", "blur_winner", "blur_type", "blur_homogeneous", "blur_motion",
    "blur_direction_ratio", "blur_method"} for a BGR patch."""
    if patch_bgr is None or patch_bgr.size == 0 or min(patch_bgr.shape[:2]) < 3:
        return {"blur_score": 100.0, "blur_winner": "homogeneous", "blur_type": "homogeneous",
                "blur_homogeneous": 100.0, "blur_motion": 0.0, "blur_direction_ratio": 1.0, "blur_method": BLUR_METHOD}
    gray = _full_gray(patch_bgr)
    energies = directional_detail(patch_bgr, gray)
    weakest = float(energies.min())
    ratio = float(energies.max() / max(weakest, 1e-6))
    homogeneous = float(np.clip((_LOG_SHARPEST - np.log10(weakest + 1e-6)) / (_LOG_SHARPEST - _LOG_BLURRIEST), 0.0, 1.0)) * 100.0
    motion = float(min(100.0, _MOTION_SCALE * (1.0 - fft_sunshine(gray))))
    return {
        "blur_score": round(max(homogeneous, motion), 1),
        "blur_winner": "motion" if motion > homogeneous else "homogeneous",
        "blur_type": "motion" if ratio >= _MOTION_RATIO else "homogeneous",
        "blur_homogeneous": round(homogeneous, 1),
        "blur_motion": round(motion, 1),
        "blur_direction_ratio": round(ratio, 1),
        "blur_method": BLUR_METHOD,
    }


def blur_score(patch_bgr):
    """Blurriness 0 (sharp) .. 100 (blurriest), on a fixed scale."""
    return blur_fields(patch_bgr)["blur_score"]


def set_blur_on_shape(shape, patch_bgr):
    """Record the patch's blurriness on its detection shape (in place)."""
    record_blur_fields(shape, blur_fields(patch_bgr))


def record_blur_fields(shape, fields):
    """Write blur fields onto a shape, dropping fields from earlier methods."""
    for stale in ("motion_streak", "blur_detail"):  # fields from earlier methods
        shape.pop(stale, None)
    shape.update(fields)


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
