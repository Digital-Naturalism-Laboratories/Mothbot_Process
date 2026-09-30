"""Check core/blur.py's blurriness against hand ratings (evaluation only; nothing is fitted).

    python tools/blur_calibration/evaluate.py [data_dir ...]

Reads sample.json + labels.json from each data_dir (default
~/.mothbot/blur_calibration; see make_sample.py and rate.py), scores every rated
patch with core.blur.blur_fields, and reports how well the score separates the
ratings overall, by the rater's reason, and by dataset, plus what each
threshold would skip. Use it to judge any change to the measure.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.blur import STREAK_RATIO, blur_fields  # noqa: E402

RATED = ("sharp", "usable", "blurry")  # "not_insect" is left out


def auc(pos, neg):
    """Chance a random positive scores higher than a random negative (ties count half)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    data_dirs = [os.path.expanduser(d) for d in (sys.argv[1:] or ["~/.mothbot/blur_calibration"])]
    rows, seen = [], set()
    for data_dir in data_dirs:
        with open(os.path.join(data_dir, "sample.json")) as f:
            sample = json.load(f)
        with open(os.path.join(data_dir, "labels.json")) as f:
            labels = json.load(f)
        for item in sample:
            label = labels.get(str(item["i"]))
            if not label or label["rating"] not in RATED or item["path"] in seen:
                continue  # unrated, not an insect, or a repeat
            image = cv2.imread(item["path"])
            if image is None:
                print(f"  missing patch, skipped: {item['path']}")
                continue
            seen.add(item["path"])
            rows.append(dict(blur_fields(image), rating=label["rating"], cause=label.get("cause") or "", dataset=item["dataset"]))

    score = np.array([r["blur_score"] for r in rows])
    rating = np.array([r["rating"] for r in rows])
    cause = np.array([r["cause"] for r in rows])
    dataset = np.array([r["dataset"] for r in rows])
    blurry = rating == "blurry"
    print(f"{len(rows)} rated patches: " + ", ".join(f"{k} {int((rating == k).sum())}" for k in RATED))
    print("median score  sharp / usable / too blurry: "
          + " / ".join(f"{np.median(score[rating == k]):.1f}" for k in RATED))
    print(f"AUC  too blurry vs rest {auc(score[blurry], score[~blurry]):.2f}   "
          f"not sharp vs sharp {auc(score[rating != 'sharp'], score[rating == 'sharp']):.2f}")
    print("AUC by reason (vs sharp+usable): " + "  ".join(
        f"{c or 'unsure'} {auc(score[blurry & (cause == c)], score[~blurry]):.2f} (n={int((blurry & (cause == c)).sum())})"
        for c in ("motion", "focus", "small", "")))
    print("AUC by dataset: " + "  ".join(
        f"{d} {auc(score[(dataset == d) & blurry], score[(dataset == d) & ~blurry]):.2f}"
        for d in sorted(set(dataset)) if ((dataset == d) & blurry).any() and ((dataset == d) & ~blurry).any()))
    streak = np.array([r["motion_streak"] for r in rows]) > STREAK_RATIO
    print(f"motion-streak rule: {int((streak & (cause == 'motion')).sum())}/{int((cause == 'motion').sum())} motion patches, "
          f"{int((streak & ~blurry).sum())}/{int((~blurry).sum())} sharp+usable patches")
    print("threshold | too blurry skipped | usable skipped | sharp skipped")
    for t in (10, 15, 20, 25, 30):
        cells = [f"{np.mean(score[rating == k] > t):.0%}" for k in ("blurry", "usable", "sharp")]
        print(f"{t:9d} | {cells[0]:>18} | {cells[1]:>14} | {cells[2]:>13}")


if __name__ == "__main__":
    main()
