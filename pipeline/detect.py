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
from datetime import datetime

from core.common import (
    find_date_folders,
    scan_for_images,
    current_timestamp,
    get_device,
    print_device_info,
)

# ~~~~Default values (used when run from CLI without args)~~~~~~~

DEFAULT_INPUT_PATH = r"G:\Shared drives\Mothbox Management\Testing\ExampleDataset\Les_BeachPalm_hopeCobo_2025-06-20\2025-06-21"
DEFAULT_YOLO_MODEL = r"..\trained_models\yolo11m_4500_imgsz1600_b1_2024-01-18\weights\yolo11m_4500_imgsz1600_b1_2024-01-18.pt"
DEFAULT_IMGSZ = 1600

# Module-level globals set by run() before processing functions are called.
# This preserves backward compatibility with existing function signatures.
YOLO_MODEL = DEFAULT_YOLO_MODEL
IMGSZ = DEFAULT_IMGSZ
DEVICE = "cpu"
GEN_BOT_DET_EVENIF_HUMAN_EXISTS = True
OVERWRITE_PREV_BOT_DETECTIONS = True
GEN_THUMBNAILS = True


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

        # PyTorch 2.6+ defaults torch.load(..., weights_only=True), which can
        # fail for trusted Ultralytics checkpoints that include model classes.
        print(
            "Retrying model load with torch.load(weights_only=False) compatibility mode..."
        )
        original_torch_load = torch.load
        original_force_weights_only = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        original_force_no_weights_only = os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        )

        def _torch_load_compat(*args, **kwargs):
            # Force unsafe-load mode for trusted local checkpoints.
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat
        try:
            # PyTorch can force weights_only=True via env var regardless of callsite.
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


def process_jpg_files(img_files, date_folder):
    """
    Processes all *.jpg files within the specified date folder, running YOLO
    detection and writing a _botdetection.json file for each image.

    Reads module globals: YOLO_MODEL, IMGSZ, DEVICE, GEN_THUMBNAILS,
    GEN_BOT_DET_EVENIF_HUMAN_EXISTS, OVERWRITE_PREV_BOT_DETECTIONS.
    """
    # Load the model
    model = load_yolo_model(YOLO_MODEL)
    model_name = os.path.basename(YOLO_MODEL)
    model_name = "Mothbot_" + model_name

    total_img_files = len(img_files)

    patch_folder_path = Path(date_folder + "/patches")
    patch_folder_path.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(img_files):

        image_path = os.path.join(date_folder, filename)
        human_json_path = os.path.join(date_folder, filename[:-4] + ".json")
        bot_json_path = os.path.join(date_folder, filename[:-4] + "_botdetection.json")

        if not is_valid_image(image_path):
            print(f"Skipping corrupt image: {image_path}")
            continue

        processed_files = idx + 1
        progress = ((processed_files - 1) / total_img_files) * 100
        print(f"({progress:.2f}%) Processing:  {filename} ")

        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            print(f"Skipping {filename}: Image file is missing or empty.")
            continue

        # Check 1: human detection file
        if os.path.isfile(human_json_path):
            print(human_json_path)
            print(
                "Earlier Human detection file exists, check to see if we should skip it"
            )
            try:
                with open(human_json_path, "r") as json_file:
                    json_data = json.load(json_file)
                    if GEN_THUMBNAILS:
                        json_data = generateThumbnailPatches_JSON(
                            image_path, json_data, patch_folder_path
                        )
                        with open(human_json_path, "w") as json_file_write:
                            json.dump(json_data, json_file_write, indent=4)
                    if not GEN_BOT_DET_EVENIF_HUMAN_EXISTS:
                        print(
                            "skipping-will not create bot detections in parallel with human detections"
                        )
                        continue
            except json.JSONDecodeError:
                print(f"error with HUMAN made {filename}: Corrupted JSON file.")

        # Check 2: existing bot detection file
        if os.path.isfile(bot_json_path):
            print(bot_json_path)
            print(
                "Earlier BOT detection file exists, check to see if we should skip it, "
            )
            try:
                with open(bot_json_path, "r") as json_file:
                    json_data = json.load(json_file)
                    if not OVERWRITE_PREV_BOT_DETECTIONS:
                        if GEN_THUMBNAILS:
                            json_data = generateThumbnailPatches_JSON(
                                image_path, json_data, patch_folder_path
                            )
                            with open(bot_json_path, "w") as json_file_write:
                                json.dump(json_data, json_file_write, indent=4)
                        print(
                            "skipping previously generated detection files that were able to be opened"
                        )
                        continue
            except json.JSONDecodeError:
                print(f"error with {filename}: Corrupted JSON file.")

        # ~~~~~~~~ Run YOLO detection ~~~~~~~~~~~~~
        try:
            print("Predict where insects are on a new image :", image_path)
            results = model.predict(
                source=image_path, imgsz=IMGSZ, device=DEVICE, verbose=False
            )
        except Exception as e:
            print(f"❌ Skipping corrupt/unreadable image: {image_path} ({e})")
            print(
                f"Skipping {filename}: Image file is missing or empty and messed up in YOLO."
            )
            continue

        # Extract OBB coordinates and crop
        shapes = []
        for result in results:
            for det_idx, obb in enumerate(result.obb.xyxyxyxy):
                points = obb.cpu().numpy().reshape((-1, 1, 2)).astype(int)
                cnt = points
                rect = cv2.minAreaRect(cnt)
                confidence = result.obb.conf[det_idx].item()

                print("rect: {}".format(rect) + "   conf: " + str(confidence))

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                center, size, angle = rect[0], rect[1], rect[2]
                pts = obb.cpu().numpy().reshape((-1, 1, 2)).astype(float)
                pts = pts.tolist()
                pts = [item for sublist in pts for item in sublist]  # flatten

                print(confidence)
                shape = {
                    "points": pts,
                    "direction": angle,
                    "score": float(confidence),
                }

                thepatchpath = ""
                if GEN_THUMBNAILS:
                    thepatchpath = generateThumbnailPatches(
                        result.orig_img, image_path, rect, det_idx, model_name
                    )
                shape["patch_path"] = thepatchpath
                shape["confidence_detection"] = confidence
                shape["identifier_bot"] = ""
                shape["identifier_human"] = ""
                shape["timestamp_detection"] = current_timestamp()
                shape["detector_bot"] = str(model_name)
                shapes.append(shape)

        image = PIL.Image.open(image_path)
        width, height = image.size

        data = {
            "version": model_name,
            "flags": {},
            "imagePath": image_path,
            "imageHeight": height,
            "imageWidth": width,
            "description": "",
            "imageData": None,
        }

        if "shapes" not in data:
            data["shapes"] = []

        for shape in shapes:
            shape_data = {
                "kie_linking": [],
                "direction": shape["direction"],
                "label": "creature",
                "score": shape["score"],
                "group_id": None,
                "description": "",
                "difficult": "false",
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "points": shape["points"],
                "patch_path": shape["patch_path"],
                "confidence_detection": shape["confidence_detection"],
                "identifier_bot": shape["identifier_bot"],
                "identifier_human": shape["identifier_human"],
                "timestamp_detection": shape["timestamp_detection"],
                "detector_bot": shape["detector_bot"],
            }
            data["shapes"].append(shape_data)

        with open(bot_json_path, "w") as f:
            json.dump(data, f, indent=4)


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
):
    """Main entry point for detection.  Called directly by the UI or via CLI."""
    global YOLO_MODEL, IMGSZ, DEVICE, GEN_THUMBNAILS
    global GEN_BOT_DET_EVENIF_HUMAN_EXISTS, OVERWRITE_PREV_BOT_DETECTIONS

    YOLO_MODEL = yolo_model or DEFAULT_YOLO_MODEL
    IMGSZ = int(imgsz)
    DEVICE = get_device()
    GEN_THUMBNAILS = gen_thumbnails
    GEN_BOT_DET_EVENIF_HUMAN_EXISTS = gen_bot_det_evenif_human_exists
    OVERWRITE_PREV_BOT_DETECTIONS = overwrite_prev_bot_detections

    print("Starting Mothbot Detection Script")
    print_device_info(selected_device=DEVICE)
    print(f"Processing {input_path} with model {YOLO_MODEL} and image size {IMGSZ}")

    date_folders = find_date_folders(input_path)
    print(f"{len(date_folders)}  nightly folders found to process")

    for date_folder_path in date_folders:
        print(date_folder_path)
        images = scan_for_images(date_folder_path)
        print(f"{len(images)}  images to process in this night: {date_folder_path}")
        process_jpg_files(images, date_folder_path)

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
    )
