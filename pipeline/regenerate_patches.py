#!/usr/bin/env python3
"""
regenerate_patches.py — Re-crop all detection patches from existing JSONs.

Quick way to refresh patch images WITHOUT re-running YOLO detection: it reuses the
detection boxes already stored in the ``_processed`` JSON files and simply re-crops
each patch from the source photo. Use it after a change to the cropping code (e.g.
the off-image black-fill fix) to replace already-written patches in place.

Unlike the Detect stage's thumbnail step, this always OVERWRITES existing patches
(skip_existing=False), so the old images are actually replaced.

Usage:
  python -m pipeline.regenerate_patches --input_path "/path/to/DatasetOrDeployment"
"""
import argparse
import json
import os
import time

from core.common import find_detection_matches_processed
from pipeline.thumbnails import generateThumbnailPatches_JSON


def run(input_path, dataset_root=None):
    """Re-crop every detection patch under *input_path* from its existing JSON.

    Parameters
    ----------
    input_path : str
        Dataset collection, deployment, or nightly folder to process.
    dataset_root : str | None
        Root of the ``_processed`` mirror tree. Defaults to *input_path*.
    """
    _dataset_root = dataset_root or input_path

    hu_pairs, bot_pairs = find_detection_matches_processed(_dataset_root, source_folder=input_path)
    # Bot and human detections both live in the _processed tree; regenerate both.
    all_pairs = [("BOT", ip, jp) for ip, jp in bot_pairs] + \
                [("HU", ip, jp) for ip, jp in hu_pairs]

    total = len(all_pairs)
    if total == 0:
        print("No detection JSONs found under _processed — nothing to regenerate.")
        return

    print(f"Regenerating patches for {total} detection file(s) — no YOLO inference, "
          f"re-cropping from existing boxes...")

    regenerated = 0
    errors = 0
    t0 = time.monotonic()

    for idx, (label, image_path, json_path) in enumerate(all_pairs, start=1):
        if not os.path.isfile(image_path):
            print(f"  ⚠️  [{idx}/{total}] source image missing, skipping: {os.path.basename(image_path)}")
            continue
        try:
            with open(json_path) as f:
                json_data = json.load(f)
            n_shapes = len(json_data.get("shapes", []))
            # Patches live flat in the same mirrored folder as the JSON.
            output_folder = os.path.dirname(json_path)
            generateThumbnailPatches_JSON(
                image_path, json_data, output_folder, skip_existing=False
            )
            regenerated += n_shapes
            if idx % 25 == 0 or idx == total:
                elapsed = time.monotonic() - t0
                rate = idx / elapsed if elapsed else 0
                eta = (total - idx) / rate if rate else 0
                print(f"  ✓ [{idx}/{total}] {label} {os.path.basename(json_path)} "
                      f"({n_shapes} patches) — ~{eta/60:.1f} min left")
        except json.JSONDecodeError:
            errors += 1
            print(f"  ❌ [{idx}/{total}] corrupt JSON: {os.path.basename(json_path)}")
        except Exception as e:
            errors += 1
            print(f"  ❌ [{idx}/{total}] {os.path.basename(json_path)}: {e}")

    print(f"\n✅ Done — re-cropped {regenerated} patches from {total} files "
          f"in {(time.monotonic() - t0)/60:.1f} min ({errors} errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True,
                        help="Dataset, deployment, or nightly folder to regenerate patches for.")
    parser.add_argument("--dataset_root", default=None,
                        help="Root of the _processed tree. Defaults to input_path.")
    args = parser.parse_args()
    run(input_path=args.input_path, dataset_root=args.dataset_root)
