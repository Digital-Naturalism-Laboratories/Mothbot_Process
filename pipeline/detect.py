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
from pipeline.thumbnails import generateThumbnailPatches_JSON
import torch
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
DELETE_OLD_MODEL_PATCHES = False  # When True, deletes patch images from the old model when switching models
GEN_HUMAN_DET_PATCHES = True    # When True, extracts patch crops from x-anylabeling annotations and writes _humandetection.json
DATASET_ROOT = None  # Set by run(); when None, outputs go next to source images (legacy)


# ~~~~Functions~~~~~~~


def load_yolo_model(model_path):
    """Load a YOLO model from a .pt or .onnx file.

    .onnx models run through ONNX Runtime, which is 2-4x faster than PyTorch
    on CPU with no other changes required. Just point the model path at the
    .onnx file — imgsz is read from the model itself and IMGSZ is updated.

    .pt models use the standard PyTorch path with a weights_only compatibility
    fallback for PyTorch 2.6+.
    """
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    if not Path(resolved_model_path).is_file():
        raise FileNotFoundError(
            f"YOLO model file not found at {resolved_model_path}. "
            "Pick a valid local .pt or .onnx file in Setup > YOLO Model Path. "
            "Mothbot does not auto-download model weights during detection."
        )

    if resolved_model_path.lower().endswith(".onnx"):
        return _load_onnx_model(resolved_model_path)

    return _load_pt_model(resolved_model_path)


def _load_onnx_model(resolved_model_path):
    """Load an ONNX model directly via ONNX Runtime (bypasses ultralytics predict).

    Why bypass ultralytics? When ultralytics loads an ONNX model it must detect
    the task type (obb vs detect) from embedded metadata. If that metadata is
    absent or misread, it applies the wrong output decoder and produces 0
    detections — even though the model is perfectly healthy in other tools
    (e.g. x-anylabeling). Running ONNX Runtime directly avoids this entirely.

    The session and input metadata are stored in module globals so the inference
    helpers (_letterbox_image, _infer_onnx_single) can use them without passing
    extra arguments through every call site.
    """
    import onnxruntime as ort
    global _EFFECTIVE_BATCH_SIZE, _IS_ONNX_MODEL
    global _onnx_session, _onnx_input_name, _onnx_imgsz, _onnx_task

    session = ort.InferenceSession(resolved_model_path, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    _onnx_input_name = inp.name
    shape = inp.shape  # [batch, channels, H, W]
    _onnx_imgsz = (int(shape[2]), int(shape[3]))

    meta = session.get_modelmeta().custom_metadata_map
    _onnx_task = meta.get("task", "obb")  # default obb; correct for this project

    _onnx_session = session
    _IS_ONNX_MODEL = True
    _EFFECTIVE_BATCH_SIZE = 1

    print(f"  ℹ️  ONNX model (task={_onnx_task}, imgsz={_onnx_imgsz}) — "
          f"running via ONNX Runtime directly (UI imgsz setting ignored).")
    print(f"  ℹ️  Batch size forced to 1 for ONNX.")
    print(f"  ✓ Loaded ONNX model (ONNX Runtime — faster CPU inference)")
    return None  # ONNX path does not use the YOLO wrapper


def _load_pt_model(resolved_model_path):
    """Load a PyTorch .pt model with a weights_only compatibility fallback."""
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


# ── ONNX Runtime inference helpers ───────────────────────────────────────────


def _letterbox_image(img, target_hw):
    """Resize + pad BGR image to (H, W) with grey fill.

    Returns (rgb_padded, scale, (pad_left, pad_top)) so the caller can map
    detection coordinates back to the original image.
    """
    h, w = img.shape[:2]
    th, tw = target_hw
    scale = min(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    pad_top = (th - nh) // 2
    pad_left = (tw - nw) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale, (pad_left, pad_top)


def _write_bot_json_data(orig_img, shapes, image_path, bot_json_path, model_name):
    """Write the bot-detection JSON for one image."""
    height, width = orig_img.shape[:2]
    data = {
        "version": model_name,
        "flags": {},
        "imagePath": image_path,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
        "imageData": None,
        "shapes": shapes,
    }
    with open(bot_json_path, "w") as f:
        json.dump(data, f, indent=2)


def _infer_onnx_single(image_path, bot_json_path, model_name, conf_thresh=0.25, iou_thresh=0.7):
    """Run ONNX Runtime on one image, write the detection JSON, return (shapes, orig_img).

    Replicates the preprocessing / postprocessing that ultralytics would do,
    but without relying on ultralytics' task-type detection (which can silently
    pick the wrong output decoder for OBB models and produce 0 detections).

    Angle unit: ultralytics OBB ONNX exports store angles in radians.
    cv2.boxPoints expects degrees, so we convert.
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        return [], None

    img_rgb, scale, (pad_left, pad_top) = _letterbox_image(orig_img, _onnx_imgsz)

    # NCHW float32, normalised 0–1
    x = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)

    raw = _onnx_session.run(None, {_onnx_input_name: x})[0]  # [1, feats, anchors]
    pred = raw[0].T  # [anchors, feats]

    is_obb = (_onnx_task == "obb")
    cx_all = pred[:, 0]
    cy_all = pred[:, 1]
    w_all  = pred[:, 2]
    h_all  = pred[:, 3]
    if is_obb:
        angle_all   = pred[:, 4]   # radians
        class_scores = pred[:, 5:]
    else:
        angle_all   = np.zeros(len(pred), dtype=np.float32)
        class_scores = pred[:, 4:]

    confs = class_scores.max(axis=1)
    keep = confs >= conf_thresh
    if not keep.any():
        _write_bot_json_data(orig_img, [], image_path, bot_json_path, model_name)
        return [], orig_img

    cx_all    = cx_all[keep]
    cy_all    = cy_all[keep]
    w_all     = w_all[keep]
    h_all     = h_all[keep]
    angle_all = angle_all[keep]
    confs     = confs[keep]

    # NMS uses top-left x,y,w,h format
    nms_rects = [
        [float(cx_all[i] - w_all[i] / 2), float(cy_all[i] - h_all[i] / 2),
         float(w_all[i]), float(h_all[i])]
        for i in range(len(cx_all))
    ]
    indices = cv2.dnn.NMSBoxes(nms_rects, confs.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        _write_bot_json_data(orig_img, [], image_path, bot_json_path, model_name)
        return [], orig_img

    indices = np.array(indices).flatten()

    filename = os.path.basename(image_path)
    stem, _, ext = filename.rpartition(".")
    shapes = []

    for det_idx, i in enumerate(indices):
        # Undo letterbox to get coords in original image pixels
        ox = (float(cx_all[i]) - pad_left) / scale
        oy = (float(cy_all[i]) - pad_top)  / scale
        ow = float(w_all[i])  / scale
        oh = float(h_all[i])  / scale
        angle_deg = float(np.degrees(float(angle_all[i])))
        conf = float(confs[i])

        pts = cv2.boxPoints(((ox, oy), (ow, oh), angle_deg))
        points = pts.tolist()
        patch_filename = f"{stem}_{det_idx}_{model_name}.{ext}" if GEN_THUMBNAILS else ""

        shapes.append({
            "kie_linking": [],
            "direction": angle_deg,
            "label": "creature",
            "score": conf,
            "group_id": None,
            "description": "",
            "difficult": "false",
            "shape_type": "rotation",
            "flags": {},
            "attributes": {},
            "points": points,
            "patch_path": patch_filename,
            "confidence_detection": conf,
            "identifier_bot": "",
            "identifier_human": "",
            "timestamp_detection": current_timestamp(),
            "detector_bot": str(model_name),
        })

    _write_bot_json_data(orig_img, shapes, image_path, bot_json_path, model_name)
    return shapes, orig_img


def is_valid_image(image_path):
    """Checks if an image file is valid using Pillow."""
    try:
        Image.open(image_path).verify()
        return True
    except (IOError, SyntaxError):
        return False


def _write_humandetection_json(source_json_data: dict, output_path: str, image_path: str):
    """Write a _humandetection.json from an x-anylabeling source JSON.

    Copies the processed shapes (which already have patch_path set by
    generateThumbnailPatches_JSON) into a new file using the same structure as
    _botdetection.json so that Classify can ingest it alongside bot runs.
    Sets version/detector_bot/identifier_bot to "HumanDetection" so Classify
    can identify the detection source.
    """
    shapes = source_json_data.get("shapes", [])
    humandetection_data = {
        "version": "HumanDetection",
        "flags": source_json_data.get("flags", {}),
        "imagePath": image_path,
        "imageHeight": source_json_data.get("imageHeight"),
        "imageWidth": source_json_data.get("imageWidth"),
        "description": "Human detection via x-anylabeling",
        "imageData": None,
        "shapes": [
            {
                **shape,
                "detector_bot": "HumanDetection",
                "identifier_bot": shape.get("identifier_bot", ""),
                "timestamp_detection": shape.get("timestamp_detection", current_timestamp()),
            }
            for shape in shapes
        ],
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(humandetection_data, f, indent=2)
    print(f"  Wrote human detection file: {os.path.basename(output_path)}")


def _resolve_output_paths(image_path, filename_stem, source_folder):
    """Return (human_json_path, bot_json_path, humandetection_json_path, patch_folder_path).

    When DATASET_ROOT is set the outputs go into the _processed mirror tree
    flat alongside the JSONs — no patches/ sub-folder.
    When DATASET_ROOT is not set, falls back to legacy behaviour (patches/
    sub-folder next to source images).
    """
    if DATASET_ROOT:
        human_json_path = get_json_output_path(image_path, "", DATASET_ROOT)
        bot_json_path = get_json_output_path(image_path, "_botdetection", DATASET_ROOT)
        humandetection_json_path = get_json_output_path(image_path, "_humandetection", DATASET_ROOT)
        # Patches go flat in the same mirrored folder as the JSONs
        patch_folder_path = Path(get_processed_folder(source_folder, DATASET_ROOT))
    else:
        # Legacy: outputs next to source image
        human_json_path = os.path.join(source_folder, filename_stem + ".json")
        bot_json_path = os.path.join(source_folder, filename_stem + "_botdetection.json")
        humandetection_json_path = os.path.join(source_folder, filename_stem + "_humandetection.json")
        patch_folder_path = Path(source_folder) / "patches"
        patch_folder_path.mkdir(parents=True, exist_ok=True)
    return human_json_path, bot_json_path, humandetection_json_path, patch_folder_path


BATCH_SIZE = 8
# Set to 1 automatically when an ONNX model is loaded (ONNX models are typically
# exported with a fixed batch size of 1 and reject larger batches).
_EFFECTIVE_BATCH_SIZE = BATCH_SIZE
# True when the loaded model is ONNX — inference bypasses ultralytics predict()
# and runs ONNX Runtime directly (avoids task-type misdetection and wrong decoders).
_IS_ONNX_MODEL = False

# ONNX Runtime session and metadata (set by _load_onnx_model).
_onnx_session = None
_onnx_input_name = "images"
_onnx_imgsz = (640, 640)   # (H, W) read from model input shape
_onnx_task = "obb"         # read from model metadata; defaults to 'obb'


def _delete_model_patches(json_data: dict, patch_folder_path) -> int:
    """Delete patch image files referenced by json_data. Returns number of files deleted."""
    deleted = 0
    for shape in json_data.get("shapes", []):
        patch_path = shape.get("patch_path", "")
        if not patch_path:
            continue
        patch_filename = os.path.basename(patch_path)
        full_path = os.path.join(str(patch_folder_path), patch_filename)
        if os.path.isfile(full_path):
            os.remove(full_path)
            deleted += 1
    return deleted


def _model_archive_path(bot_json_path: str, model_name: str) -> str:
    """Return the deterministic archive path for a given model's botdetection JSON.

    Strips the .pt extension from model_name to avoid double extensions like .pt.json.
    Example: img_botdetection.json + "Mothbot_yolo11m_v1.pt" → img_botdetection_Mothbot_yolo11m_v1.json
    """
    slug = model_name.removesuffix(".pt").replace(" ", "_")
    stem = bot_json_path[: -len("_botdetection.json")]
    return f"{stem}_botdetection_{slug}.json"


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


def _crop_obb_fast(img, points):
    """Crop an OBB detection using only the local image region.

    Rotating the full 1600px image for every detection is very slow.
    This extracts a small padded ROI around the bounding box first,
    rotates only that region, then crops the patch — roughly 100x less work.
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    w, h = max(1, int(size[0])), max(1, int(size[1]))

    # Padding large enough to avoid clipping after rotation
    pad = int(max(w, h) * 0.6) + 4
    ih, iw = img.shape[:2]
    x1 = max(0, int(center[0]) - w // 2 - pad)
    y1 = max(0, int(center[1]) - h // 2 - pad)
    x2 = min(iw, int(center[0]) + w // 2 + pad)
    y2 = min(ih, int(center[1]) + h // 2 + pad)

    roi = img[y1:y2, x1:x2]
    local_center = (center[0] - x1, center[1] - y1)
    rh, rw = roi.shape[:2]
    M = cv2.getRotationMatrix2D(local_center, angle, 1)
    roi_rot = cv2.warpAffine(roi, M, (rw, rh), flags=cv2.INTER_LINEAR)
    patch = cv2.getRectSubPix(roi_rot, (w, h), local_center)
    return patch


def _write_patches_for_image(orig_img, shapes, patch_folder_path):
    """Extract and write patch images for all detections in one photo.

    Designed to run in a worker thread while the main thread runs the next
    YOLO batch. Returns list of written paths for preview emission.
    """
    written = []
    patch_folder_path = Path(patch_folder_path)
    for shape in shapes:
        patch_filename = shape.get("patch_path", "")
        if not patch_filename:
            continue
        try:
            patch = _crop_obb_fast(orig_img, shape["points"])
            if patch is not None and patch.size > 0:
                out_path = patch_folder_path / patch_filename
                cv2.imwrite(str(out_path), patch)
                written.append(str(out_path))
        except Exception as e:
            print(f"  ⚠️  patch crop failed for {patch_filename}: {e}")
    return written


# Number of worker threads for concurrent patch writing.
# IO-bound work — 4 threads is a good default for SSD + CPU crop.
PATCH_WORKERS = min(4, os.cpu_count() or 2)


def _save_result(result, image_path, bot_json_path, model_name):
    """Extract OBBs, write the detection JSON, and return shape data.

    Patch images are NOT written here — the caller submits
    _write_patches_for_image() to a thread pool so patch IO overlaps
    with the next YOLO batch. Patch filenames are pre-computed and stored
    in the JSON so the file is immediately complete and readable.

    Returns (shapes, orig_img).
    """
    shapes = []
    orig_img = result.orig_img

    if result.obb is not None:
        filename = os.path.basename(image_path)
        stem, _, ext = filename.rpartition(".")

        for det_idx, obb in enumerate(result.obb.xyxyxyxy):
            pts = obb.cpu().numpy().reshape(-1, 2)
            rect = cv2.minAreaRect(pts.astype(np.float32))
            confidence = result.obb.conf[det_idx].item()
            _, _, angle = rect

            points = pts.tolist()
            patch_filename = f"{stem}_{det_idx}_{model_name}.{ext}" if GEN_THUMBNAILS else ""

            shapes.append({
                "kie_linking": [],
                "direction": angle,
                "label": "creature",
                "score": float(confidence),
                "group_id": None,
                "description": "",
                "difficult": "false",
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "points": points,
                "patch_path": patch_filename,
                "confidence_detection": confidence,
                "identifier_bot": "",
                "identifier_human": "",
                "timestamp_detection": current_timestamp(),
                "detector_bot": str(model_name),
            })

    height, width = orig_img.shape[:2]
    data = {
        "version": model_name,
        "flags": {},
        "imagePath": image_path,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
        "imageData": None,
        "shapes": shapes,
    }
    with open(bot_json_path, "w") as f:
        json.dump(data, f, indent=2)

    return shapes, orig_img


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

    global _EFFECTIVE_BATCH_SIZE, _IS_ONNX_MODEL
    _EFFECTIVE_BATCH_SIZE = BATCH_SIZE  # reset; _load_onnx_model may lower this to 1
    _IS_ONNX_MODEL = False              # reset; _load_onnx_model may set this to True
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

        human_json_path, bot_json_path, humandetection_json_path, patch_folder_path = _resolve_output_paths(
            image_path, filename_stem, source_folder
        )

        print(f"({(idx / total) * 100:.1f}%) Screening: {filename}")

        if not is_valid_image(image_path):
            print(f"Skipping corrupt image: {image_path}")
            continue
        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            print(f"Skipping {filename}: Image file is missing or empty.")
            continue

        # Check 1: human detection file (x-anylabeling annotation)
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
                if GEN_HUMAN_DET_PATCHES:
                    _write_humandetection_json(json_data, humandetection_json_path, image_path)
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

                # Overwrite is enabled — check if the existing JSON was made with a different model.
                old_version = json_data.get("version", "")
                if old_version and old_version != model_name:
                    current_model_archive = _model_archive_path(bot_json_path, model_name)
                    old_model_archive = _model_archive_path(bot_json_path, old_version)
                    if os.path.isfile(current_model_archive):
                        # We've run this model before — restore it instead of re-inferring.
                        os.rename(bot_json_path, old_model_archive)
                        os.rename(current_model_archive, bot_json_path)
                        print(f"Restored archived detections for {model_name} (archived current → {os.path.basename(old_model_archive)})")
                        if GEN_THUMBNAILS:
                            with open(bot_json_path, "r") as f:
                                restored_data = json.load(f)
                            restored_data = generateThumbnailPatches_JSON(image_path, restored_data, patch_folder_path)
                            with open(bot_json_path, "w") as f:
                                json.dump(restored_data, f, indent=4)
                        continue  # skip YOLO inference
                    else:
                        # Genuinely new model — archive the old run and proceed to inference.
                        if DELETE_OLD_MODEL_PATCHES:
                            n = _delete_model_patches(json_data, patch_folder_path)
                            print(f"Deleted {n} patch file(s) from old model ({old_version})")
                        os.rename(bot_json_path, old_model_archive)
                        print(f"Archived old detections ({old_version}) → {os.path.basename(old_model_archive)}")

            except json.JSONDecodeError:
                print(f"error with {filename}: Corrupted JSON file.")

        pending.append((image_path, bot_json_path, patch_folder_path))

    if not pending:
        print("No images need YOLO inference.")
        return

    print(f"\nRunning YOLO on {len(pending)} image(s) in batches of up to {_EFFECTIVE_BATCH_SIZE}...")
    print(f"Patch writing: {PATCH_WORKERS} worker thread(s) (runs concurrently with inference)")

    # ── Phase 2: batch inference + concurrent patch writing ───────────────────
    # Strategy: YOLO inference runs on the main thread. After each batch, JSON
    # files are written immediately (fast). Patch image extraction/IO is
    # submitted to a thread pool so it overlaps with the next YOLO batch rather
    # than blocking it.
    images_done = 0
    total_pending = len(pending)
    infer_start = time.monotonic()
    patch_futures = []  # (future, patch_folder_path, filename)

    with ThreadPoolExecutor(max_workers=PATCH_WORKERS) as executor:
        for batch_start in range(0, len(pending), _EFFECTIVE_BATCH_SIZE):
            batch = pending[batch_start: batch_start + _EFFECTIVE_BATCH_SIZE]
            batch_paths = [item[0] for item in batch]

            print(f"  Batch {batch_start // _EFFECTIVE_BATCH_SIZE + 1}: predicting {len(batch)} image(s)...")

            # Collect (image_path, bot_json_path, patch_folder_path, shapes, orig_img)
            # shapes=None means the image failed; skip patch writing for that entry.
            batch_outcomes = []

            if _IS_ONNX_MODEL:
                # Direct ONNX Runtime path — batch is always 1 for ONNX models.
                image_path, bot_json_path, patch_folder_path = batch[0]
                try:
                    shapes, orig_img = _infer_onnx_single(
                        image_path, bot_json_path, model_name
                    )
                    batch_outcomes.append(
                        (image_path, bot_json_path, patch_folder_path, shapes, orig_img)
                    )
                except Exception as e:
                    print(f"❌ Skipping {os.path.basename(image_path)}: {e}")
                    batch_outcomes.append(
                        (image_path, bot_json_path, patch_folder_path, None, None)
                    )
            else:
                # ultralytics predict path (PT models).
                _pkw = {"source": batch_paths, "device": DEVICE, "verbose": False, "imgsz": IMGSZ, "max_det": 1000}
                try:
                    batch_results = model.predict(**_pkw)
                except Exception as e:
                    print(f"⚠️  Batch failed ({e}), retrying one image at a time.")
                    batch_results = []
                    for img_path, _, _ in batch:
                        try:
                            res = model.predict(**{**_pkw, "source": img_path})
                            batch_results.append(res[0])
                        except Exception as e2:
                            print(f"❌ Skipping {os.path.basename(img_path)}: {e2}")
                            batch_results.append(None)

                for result, (image_path, bot_json_path, patch_folder_path) in zip(batch_results, batch):
                    if result is None:
                        batch_outcomes.append(
                            (image_path, bot_json_path, patch_folder_path, None, None)
                        )
                        continue
                    try:
                        shapes, orig_img = _save_result(
                            result, image_path, bot_json_path, model_name
                        )
                        batch_outcomes.append(
                            (image_path, bot_json_path, patch_folder_path, shapes, orig_img)
                        )
                    except Exception as e:
                        print(f"❌ Error saving results for {os.path.basename(image_path)}: {e}")
                        batch_outcomes.append(
                            (image_path, bot_json_path, patch_folder_path, None, None)
                        )

            # Shared: schedule patch writing and emit progress for this batch.
            for image_path, bot_json_path, patch_folder_path, shapes, orig_img in batch_outcomes:
                filename = os.path.basename(image_path)
                if shapes is None:
                    images_done += 1
                    continue

                if GEN_THUMBNAILS and shapes:
                    future = executor.submit(
                        _write_patches_for_image, orig_img.copy(), shapes, patch_folder_path
                    )
                    patch_futures.append((future, patch_folder_path, filename))

                images_done += 1
                elapsed = time.monotonic() - infer_start
                avg = elapsed / images_done
                eta_secs = avg * (total_pending - images_done)
                eta_str = _format_eta(eta_secs) if images_done < total_pending else "done"
                print(f"  ✓ {filename}: {len(shapes)} detection(s) — "
                      f"{images_done}/{total_pending} images — ETA {eta_str}")

            # After each batch, emit previews for patch jobs already finished.
            # Workers run concurrently with inference, so many patches are done
            # by the time we reach here — no need to wait until the very end.
            # We emit the last written patch per image (one Gradio update per image).
            still_pending = []
            for fut, pf_path, fname in patch_futures:
                if fut.done():
                    try:
                        paths = fut.result()
                        if paths:
                            emit_preview(paths[-1])
                    except Exception as e:
                        print(f"❌ Patch write error for {fname}: {e}")
                else:
                    still_pending.append((fut, pf_path, fname))
            patch_futures = still_pending

        # Collect any patch jobs that finished after the last batch completed.
        if patch_futures:
            print(f"\n  Finishing {len(patch_futures)} patch write job(s)...")
        for future, patch_folder_path, filename in patch_futures:
            try:
                paths = future.result()
                if paths:
                    emit_preview(paths[-1])
            except Exception as e:
                print(f"❌ Patch write error for {filename}: {e}")


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
    gen_human_det_patches=True,
    delete_old_model_patches=False,
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
    global GEN_BOT_DET_EVENIF_HUMAN_EXISTS, OVERWRITE_PREV_BOT_DETECTIONS
    global GEN_HUMAN_DET_PATCHES, DELETE_OLD_MODEL_PATCHES, DATASET_ROOT

    YOLO_MODEL = yolo_model or DEFAULT_YOLO_MODEL
    IMGSZ = int(imgsz)
    DEVICE = get_device()
    GEN_THUMBNAILS = gen_thumbnails
    GEN_BOT_DET_EVENIF_HUMAN_EXISTS = gen_bot_det_evenif_human_exists
    OVERWRITE_PREV_BOT_DETECTIONS = overwrite_prev_bot_detections
    GEN_HUMAN_DET_PATCHES = gen_human_det_patches
    DELETE_OLD_MODEL_PATCHES = delete_old_model_patches
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
    parser.add_argument("--gen_human_det_patches", default=True, required=False)
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
        gen_human_det_patches=(
            bool(int(args.gen_human_det_patches))
            if not isinstance(args.gen_human_det_patches, bool)
            else args.gen_human_det_patches
        ),
        dataset_root=args.dataset_root,
    )
