"""Build a patch sample to rate for blurriness.

    python tools/blur_calibration/make_sample.py <projects_folder> [data_dir]

Takes ~20 patches from every dataset under projects_folder that has a
_processed folder, spread evenly over the current blurriness score and patch
size, plus 15 repeats in the last third (to measure how consistent the ratings
are). Writes <data_dir>/sample.json; data_dir defaults to
~/.mothbot/blur_calibration. Refuses to overwrite an existing sample, because
labels.json refers to it by position.
"""
import json
import os
import random
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.blur import blur_score  # noqa: E402

PER_DATASET, POOL, REPEATS = 20, 300, 15


def main():
    projects = os.path.expanduser(sys.argv[1])
    data_dir = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/.mothbot/blur_calibration")
    out_path = os.path.join(data_dir, "sample.json")
    if os.path.exists(out_path):
        sys.exit(f"{out_path} already exists (its ratings depend on it). Use a new data_dir.")
    rng = random.Random(7)
    items = []
    for dataset in sorted(os.listdir(projects)):
        base = os.path.join(projects, dataset)
        if not os.path.isdir(base):
            continue
        paths = []
        for dirpath, _dirnames, filenames in os.walk(base):
            if "_processed" in dirpath.split(os.sep):
                paths += [os.path.join(dirpath, f) for f in filenames if "_Mothbot_" in f and f.lower().endswith(".jpg")]
        if not paths:
            continue
        pool = []
        for path in rng.sample(paths, min(POOL, len(paths))):
            image = cv2.imread(path)
            if image is None or min(image.shape[:2]) < 8:
                continue
            pool.append({"path": path, "dataset": dataset, "w": image.shape[1], "h": image.shape[0], "score": blur_score(image)})
        pool.sort(key=lambda r: r["score"])
        per_bin = PER_DATASET // 5
        for i in range(5):  # score quintiles, each spread across patch size
            bin_rows = sorted(pool[i * len(pool) // 5:(i + 1) * len(pool) // 5], key=lambda r: r["w"] * r["h"])
            if len(bin_rows) < per_bin:
                items += bin_rows
            else:
                items += [bin_rows[int(k * (len(bin_rows) - 1) / max(1, per_bin - 1))] for k in range(per_bin)]
        print(f"{dataset}: {len(paths)} patches")

    seen, unique = set(), []
    for r in items:
        if r["path"] not in seen:
            seen.add(r["path"])
            unique.append(r)
    rng.shuffle(unique)
    order = unique[:]
    for r in rng.sample(unique[: len(unique) // 2], min(REPEATS, len(unique) // 2)):
        order.insert(rng.randint(len(order) * 2 // 3, len(order)), dict(r, repeat_of=r["path"]))
    for i, r in enumerate(order):
        r["i"] = i
    os.makedirs(data_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(order, f, indent=1)
    print(f"Wrote {len(order)} items ({len(unique)} unique) to {out_path}")


if __name__ == "__main__":
    main()
