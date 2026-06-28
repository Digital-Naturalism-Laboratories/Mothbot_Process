#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import numpy as np
import os
import re
import json
import PIL.Image
from pathlib import Path
import argparse
from PIL import Image  # For image format verification
from pipeline.thumbnails import generateThumbnailPatches, generateThumbnailPatches_JSON
import torch
import time
from datetime import datetime

from core.preview import emit_preview
from core.common import (
    find_date_folders,
    find_images_recursive,
    scan_for_images,
    current_timestamp,
    get_device,
    print_device_info,
)
from core.paths import (
    get_processed_folder,
    get_json_output_path,
    find_processed_json,
)

# ~~~~Default values (used when run from CLI without args)~~~~~~~

DEFAULT_INPUT_PATH = r"G:\Shared drives\Mothbox Management\Testing\ExampleDataset\Les_BeachPalm_hopeCobo_2025-06-20\2025-06-21"
DEFAULT_YOLO_MODEL = r"..\trained_models\yolo11m_4500_imgsz1600_b1_2024-01-18\weights\yolo11m_4500_imgsz1600_b1_2024-01-18.pt"
DEFAULT_IMGSZ = 1600

# Module-level globals set by run() before processing functions are called.
YOLO_MODEL = DEFAULT_YOLO_MODEL
IMGSZ = DEFAULT_IMGSZ
DEVICE = "cpu"
GEN_BOT_DET_EVENIF_HUMAN_EXISTS = True
OVERWRITE_PREV_BOT_DETECTIONS = True
GEN_THUMBNAILS = True
DATASET_ROOT = None  # Set by run(); when None, outputs go next to source images (legacy)


# ~~~~Functions~~~~~~~


def load_yolo_model(model_path):
    """Load YOLO model with a PyTorch 2.6 compatibility fallback."""
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    if not Path(resolved_model_path).is_file():
        raise FileNotFoundError(
            "YOLO model file not found at "
            f"{resolved_model_path}. "
            "Pick a valid local .pt file in Setup > YOLO Model Path. "
            "Mothbot does not auto-download model weights during detection."
        )

    try:
        return YOLO(resolved_model_path)
    except Exception as err:
        message = str(err)
        if "Weights only load failed" not in message:
            raise

        print(
            "Retrying model load with torch.load(weights_only=False) compatibility mode..."
        )
        original_torch_load = torch.load
        original_force_weights_only = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        original_force_no_weights_only = os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        )

        def _torch_load_compat(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat
        try:
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            return YOLO(resolved_model_path)
        finally:
            torch.load = original_torch_load
            if original_force_weights_only is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = (
                    original_force_weights_only
                )
            if original_force_no_weights_only is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = (
                    original_force_no_weights_only
                )


def is_valid_image(image_path):
    """Checks if an image file is valid using Pillow."""
    try:
        Image.open(image_path).verify()
        return True
    except (IOError, SyntaxError):
        return False


def _resolve_output_paths(image_path, filename_stem, source_folder):
    """Return (human_json_path, bot_json_path, patch_folder_path).

    When DATASET_ROOT is set the outputs go into the _processed mirror tree
    flat alongside the JSONs — no patches/ sub-folder.
    When DATASET_ROOT is not set, falls back to legacy behaviour (patches/
    sub-folder next to source images).
    """
    if DATASET_ROOT:
        human_json_path = get_json_output_path(image_path, "", DATASET_ROOT)
        bot_json_path = get_json_output_path(image_path, "_botdetection", DATASET_ROOT)
        # Patches go flat in the same mirrored folder as the JSONs
        patch_folder_path = Path(get_processed_folder(source_folder, DATASET_ROOT))
    else:
        # Legacy: outputs next to source image
        human_json_path = os.path.join(source_folder, filename_stem + ".json")
        bot_json_path = os.path.join(source_folder, filename_stem + "_botdetection.json")
        patch_folder_path = Path(source_folder) / "patches"
        patch_folder_path.mkdir(parents=True, exist_ok=True)
    return human_json_path, bot_json_path, patch_folder_path


BATCH_SIZE = 8


def _format_eta(seconds: float) -> str:
    """Return a human-readable ETA string, e.g. '2h 4m', '3m 12s', '45s'."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _save_result(result, image_path, bot_json_path, patch_folder_path, model_name):
    """Extract OBBs from one YOLO result, save patches, and write the JSON.

    Returns the list of shape dicts (may be empty if no detections).
    """
    shapes = []
    if result.obb is None:
        pass
    else:
        for det_idx, obb in enumerate(result.obb.xyxyxyxy):
            points = obb.cpu().numpy().reshape((-1, 1, 2)).astype(int)
            rect = cv2.minAreaRect(points)
            confidence = result.obb.conf[det_idx].item()

            print("rect: {}".format(rect) + "   conf: " + str(confidence))

            center, size, angle = rect[0], rect[1], rect[2]
            pts = obb.cpu().numpy().reshape((-1, 1, 2)).astype(float)
            pts = pts.tolist()
            pts = [item for sublist in pts for item in sublist]

            thepatchpath = ""
            if GEN_THUMBNAILS:
                thepatchpath = generateThumbnailPatches(
                    result.orig_img, image_path, rect, det_idx, model_name,
                    patch_folder=str(patch_folder_path),
                )

            shapes.append({
                "points": pts,
                "direction": angle,
                "score": float(confidence),
                "patch_path": thepatchpath,
                "confidence_detection": confidence,
                "identifier_bot": "",
                "identifier_human": "",
                "timestamp_detection": current_timestamp(),
                "detector_bot": str(model_name),
            })

    # Use orig_img shape for width/height — avoids a redundant PIL disk read.
    height, width = result.orig_img.shape[:2]

    data = {
        "version": model_name,
        "flags": {},
        "imagePath": image_path,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
        "imageData": None,
        "shapes": [
            {
                "kie_linking": [],
                "direction": s["direction"],
                "label": "creature",
                "score": s["score"],
                "group_id": None,
                "description": "",
                "difficult": "false",
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "points": s["points"],
                "patch_path": s["patch_path"],
                "confidence_detection": s["confidence_detection"],
                "identifier_bot": s["identifier_bot"],
                "identifier_human": s["identifier_human"],
                "timestamp_detection": s["timestamp_detection"],
                "detector_bot": s["detector_bot"],
            }
            for s in shapes
        ],
    }
    with open(bot_json_path, "w") as f:
        json.dump(data, f, indent=4)

    return shapes


def process_image_list(img_files, dataset_root=None):
    """Process a flat list of absolute image paths.

    Phase 1 — pre-screening: validate each image and handle existing JSON files
    (skip or regenerate thumbnails as needed).  Builds a list of images that
    actually need YOLO inference.

    Phase 2 — batch inference: run YOLO on groups of BATCH_SIZE images at once
    for better GPU utilisation.  Falls back to single-image mode automatically
    if a batch fails (e.g. GPU OOM).
    """
    global DATASET_ROOT
    DATASET_ROOT = dataset_root

    model = load_yolo_model(YOLO_MODEL)
    model_name = "Mothbot_" + os.path.basename(YOLO_MODEL)

    total = len(img_files)

    # ── Phase 1: pre-screening ────────────────────────────────────────────────
    # pending = list of (image_path, bot_json_path, patch_folder_path)
    pending = []

    for idx, image_path in enumerate(img_files):
        filename = os.path.basename(image_path)
        filename_stem = filename[:-4] if filename.lower().endswith(".jpg") else filename
        source_folder = os.path.dirname(image_path)

        human_json_path, bot_json_path, patch_folder_path = _resolve_output_paths(
            image_path, filename_stem, source_folder
        )

        print(f"({(idx / total) * 100:.1f}%) Screening: {filename}")

        if not is_valid_image(image_path):
            print(f"Skipping corrupt image: {image_path}")
            continue
        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            print(f"Skipping {filename}: Image file is missing or empty.")
            continue

        # Check 1: human detection file
        human_json_source = os.path.join(source_folder, filename_stem + ".json")
        effective_human_json = human_json_path if os.path.isfile(human_json_path) else (
            human_json_source if os.path.isfile(human_json_source) else None
        )
        if effective_human_json:
            print(effective_human_json)
            print("Earlier Human detection file exists, check to see if we should skip it")
            try:
                with open(effective_human_json, "r") as f:
                    json_data = json.load(f)
                if GEN_THUMBNAILS:
                    json_data = generateThumbnailPatches_JSON(image_path, json_data, patch_folder_path)
                    with open(human_json_path, "w") as f:
                        json.dump(json_data, f, indent=4)
                if not GEN_BOT_DET_EVENIF_HUMAN_EXISTS:
                    print("skipping-will not create bot detections in parallel with human detections")
                    continue
            except json.JSONDecodeError:
                print(f"error with HUMAN made {filename}: Corrupted JSON file.")

        # Check 2: existing bot detection file
        if os.path.isfile(bot_json_path):
            print(bot_json_path)
            print("Earlier BOT detection file exists, check to see if we should skip it, ")
            try:
                with open(bot_json_path, "r") as f:
                    json_data = json.load(f)
                if not OVERWRITE_PREV_BOT_DETECTIONS:
                    if GEN_THUMBNAILS:
                        json_data = generateThumbnailPatches_JSON(image_path, json_data, patch_folder_path)
                        with open(bot_json_path, "w") as f:
                            json.dump(json_data, f, indent=4)
                    print("skipping previously generated detection files that were able to be opened")
                    continue
            except json.JSONDecodeError:
                print(f"error with {filename}: Corrupted JSON file.")

        pending.append((image_path, bot_json_path, patch_folder_path))

    if not pending:
        print("No images need YOLO inference.")
        return

    print(f"\nRunning YOLO on {len(pending)} image(s) in batches of up to {BATCH_SIZE}...")

    # ── Phase 2: batch inference ──────────────────────────────────────────────
    images_done = 0
    total_pending = len(pending)
    infer_start = time.monotonic()

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        batch_paths = [item[0] for item in batch]

        print(f"  Batch {batch_start // BATCH_SIZE + 1}: predicting {len(batch)} image(s)...")
        try:
            batch_results = model.predict(
                source=batch_paths, imgsz=IMGSZ, device=DEVICE, verbose=False
            )
        except Exception as e:
            print(f"⚠️  Batch failed ({e}), retrying one image at a time.")
            batch_results = []
            for image_path, _, _ in batch:
                try:
                    res = model.predict(source=image_path, imgsz=IMGSZ, device=DEVICE, verbose=False)
                    batch_results.append(res[0])
                except Exception as e2:
                    print(f"❌ Skipping {os.path.basename(image_path)}: {e2}")
                    batch_results.append(None)

        for result, (image_path, bot_json_path, patch_folder_path) in zip(batch_results, batch):
            if result is None:
                images_done += 1
                continue
            filename = os.path.basename(image_path)
            try:
                shapes = _save_result(result, image_path, bot_json_path, patch_folder_path, model_name)
            except Exception as e:
                print(f"❌ Error saving results for {filename}: {e}")
                images_done += 1
                continue

            images_done += 1
            elapsed = time.monotonic() - infer_start
            avg = elapsed / images_done
            eta_secs = avg * (total_pending - images_done)
            eta_str = _format_eta(eta_secs) if images_done < total_pending else "done"
            # Emit previews before printing so they're queued when Gradio yields on the print.
            if GEN_THUMBNAILS and shapes:
                for shape in shapes:
                    patch_path = shape.get("patch_path", "")
                    if patch_path:
                        emit_preview(str(patch_folder_path / os.path.basename(patch_path)))

            print(f"  ✓ {filename}: {len(shapes)} detection(s) — "
                  f"{images_done}/{total_pending} images — ETA {eta_str}")


def process_jpg_files(img_files, date_folder):
    """Legacy per-folder entry point.  Still used when run() is called without
    a dataset_root (e.g. the old nightly-folder workflow).
    """
    process_image_list(
        [os.path.join(date_folder, f) if not os.path.isabs(f) else f for f in img_files],
        dataset_root=DATASET_ROOT,
    )


def crop_rect_old(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot


def crop_rect(img, rect, interpolation=cv2.INTER_LINEAR):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height), flags=interpolation)
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot


# ---------------------------------------------------------------------------
# run() – callable from the Gradio UI (no subprocess needed)
# ---------------------------------------------------------------------------


def run(
    input_path,
    yolo_model=None,
    imgsz=DEFAULT_IMGSZ,
    overwrite_prev_bot_detections=True,
    gen_bot_det_evenif_human_exists=True,
    gen_thumbnails=True,
    dataset_root=None,
):
    """Main entry point for detection.  Called directly by the UI or via CLI.

    Parameters
    ----------
    input_path : str
        The folder the user selected to process.  This can be:
        - A top-level dataset collection folder (contains deployment sub-folders)
        - A single deployment folder
        - A single nightly folder
        Structure is discovered automatically; all .jpg files found under
        *input_path* (excluding _processed/ and patches/ sub-trees) are processed.
    dataset_root : str | None
        If provided, outputs go into ``<dataset_root>/_processed/``.
        Defaults to *input_path* itself (so ``_processed/`` is created inside
        the chosen folder).
    """
    global YOLO_MODEL, IMGSZ, DEVICE, GEN_THUMBNAILS
    global GEN_BOT_DET_EVENIF_HUMAN_EXISTS, OVERWRITE_PREV_BOT_DETECTIONS, DATASET_ROOT

    YOLO_MODEL = yolo_model or DEFAULT_YOLO_MODEL
    IMGSZ = int(imgsz)
    DEVICE = get_device()
    GEN_THUMBNAILS = gen_thumbnails
    GEN_BOT_DET_EVENIF_HUMAN_EXISTS = gen_bot_det_evenif_human_exists
    OVERWRITE_PREV_BOT_DETECTIONS = overwrite_prev_bot_detections
    DATASET_ROOT = dataset_root or input_path

    print("Starting Mothbot Detection Script")
    print_device_info(selected_device=DEVICE)
    print(f"Processing {input_path} with model {YOLO_MODEL} and image size {IMGSZ}")
    print(f"Outputs will be written to: {DATASET_ROOT}/_processed/")

    images = find_images_recursive(input_path)
    print(f"{len(images)} images found to process")

    process_image_list(images, dataset_root=DATASET_ROOT)

    print("Finished Running Detections!")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, required=False)
    parser.add_argument("--yolo_model", default=DEFAULT_YOLO_MODEL, required=False)
    parser.add_argument("--imgsz", default=DEFAULT_IMGSZ, type=int, required=False)
    parser.add_argument(
        "--gen_bot_det_evenif_human_exists", default=True, required=False
    )
    parser.add_argument("--overwrite_prev_bot_detections", default=True, required=False)
    parser.add_argument("--gen_thumbnails", default=True, required=False)
    parser.add_argument(
        "--dataset_root",
        default=None,
        required=False,
        help="Root folder for _processed output tree. Defaults to input_path.",
    )
    args = parser.parse_args()

    run(
        input_path=args.input_path,
        yolo_model=args.yolo_model,
        imgsz=args.imgsz,
        overwrite_prev_bot_detections=bool(int(args.overwrite_prev_bot_detections)),
        gen_bot_det_evenif_human_exists=bool(int(args.gen_bot_det_evenif_human_exists)),
        gen_thumbnails=(
            bool(int(args.gen_thumbnails))
            if not isinstance(args.gen_thumbnails, bool)
            else args.gen_thumbnails
        ),
        dataset_root=args.dataset_root,
    )
