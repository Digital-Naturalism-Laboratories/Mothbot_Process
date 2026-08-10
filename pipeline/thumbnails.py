#!/usr/bin/env python3

import json
import os
from pathlib import Path
import cv2
import numpy as np

from core.common import find_date_folders, scan_for_images

INPUT_PATH = r"F:\Panama\PEA_PeaPorch_AdeptTurca_2024-09-01\2024-09-01"


def crop_rect(
    img, rect, interpolation=cv2.INTER_LINEAR
):  # cv2.INTER_LANCZOS4  cv2.INTER_LINEAR cv2.INTER_CUBIC
    center, size, angle = rect[0], rect[1], rect[2]
    center = tuple(map(float, center))
    w, h = int(size[0]), int(size[1])
    # Rotate and crop to the w×h patch in a single warp with a black constant
    # border, so a detection box that extends past the source-image edge yields
    # solid black there instead of cv2.getRectSubPix's replicated-edge colour
    # streak (which would confuse the downstream ID model).
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # (w-1)/2, (h-1)/2 matches the framing cv2.getRectSubPix used before, so
    # in-bounds patches are unchanged.
    M[0, 2] += (w - 1) / 2 - center[0]
    M[1, 2] += (h - 1) / 2 - center[1]
    img_crop = cv2.warpAffine(
        img, M, (w, h), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return img_crop


# TODO - save patch_img width and height along with file path
def generateThumbnailPatches_JSON(
    image_path, json_data, output_folder, skip_existing=True
):
    """Crop detections from *image_path* and write patches into *output_folder*.

    Patches are saved flat inside *output_folder* — no ``patches/`` sub-folder.
    The ``patch_path`` stored in each JSON shape is updated to just the
    filename (no path prefix).

    Parameters
    ----------
    image_path : str
        Absolute path to the source .jpg image.
    json_data : dict
        Parsed detection JSON (modified in-place and returned).
    output_folder : str | Path
        Folder where patch images are written.  When processing with the new
        ``_processed`` layout this is the mirrored folder for the source image.
        For legacy use it can still be a ``patches/`` sub-folder — the function
        doesn't care either way.
    skip_existing : bool
        Skip writing a patch if the file already exists.
    """
    model_name = json_data.get("version")
    if not model_name.startswith("Mothbot"):
        model_name = "HumanDetection"

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    loaded_images = {}
    updated_shapes = []

    for shape_index, shape in enumerate(json_data["shapes"]):
        filename = os.path.basename(image_path)
        patchfilename = f"{filename.split('.')[0]}_{shape_index}_{model_name}.{filename.split('.')[1]}"
        patchfullpath = output_folder / patchfilename

        # Store just the filename — no "patches/" prefix
        shape["patch_path"] = patchfilename

        if os.path.exists(patchfullpath) and skip_existing:
            print("Thumbnail exists, skipping")
        else:
            if image_path not in loaded_images:
                loaded_images[image_path] = cv2.imread(image_path)
            image = loaded_images[image_path]
            points = np.array(shape["points"], dtype=np.float32)
            rect = cv2.minAreaRect(points)
            img_crop = crop_rect(image, rect)
            cv2.imwrite(str(patchfullpath), img_crop)

        updated_shapes.append(shape)

    json_data["shapes"] = updated_shapes
    loaded_images.clear()
    return json_data


def generateThumbnailPatches(img, thefilepath, rectangle, detnum, modelname, patch_folder=None):
    """Crop a detection bounding box and save it as a thumbnail patch.

    Parameters
    ----------
    img : numpy.ndarray
        Source image array (as returned by OpenCV / YOLO orig_img).
    thefilepath : str
        Absolute path to the source .jpg image.
    rectangle : tuple
        OpenCV minAreaRect result.
    detnum : int
        Detection index within this image (used in the patch filename).
    modelname : str
        Name of the YOLO model (used in the patch filename).
    patch_folder : str | None
        Absolute path to the folder where the patch should be saved.
        When None, falls back to ``<image_dir>/patches/`` (legacy behaviour).

    Returns
    -------
    str
        The patch filename (no path prefix) — stored as ``patch_path`` in JSON.
        Legacy callers that passed ``patch_folder=None`` receive the old
        ``"patches/<filename>"`` format so existing JSON files stay consistent.
    """
    filename = os.path.basename(thefilepath)
    patchfilename = (
        filename.split(".")[0]
        + "_"
        + str(detnum)
        + "_"
        + modelname
        + "."
        + filename.split(".")[1]
    )

    if patch_folder is None:
        # Legacy: write into <image_dir>/patches/ and keep the old relative path format
        directory_path = os.path.dirname(thefilepath)
        patch_folder_path = Path(directory_path + "/patches")
        patch_folder_path.mkdir(parents=True, exist_ok=True)
        patchfullpath = patch_folder_path / patchfilename
        cv2.imwrite(str(patchfullpath), crop_rect(img, rectangle))
        return f"patches/{patchfilename}"
    else:
        # New _processed layout: write flat into the given folder, return bare filename
        patch_folder_path = Path(patch_folder)
        patch_folder_path.mkdir(parents=True, exist_ok=True)
        patchfullpath = patch_folder_path / patchfilename
        cv2.imwrite(str(patchfullpath), crop_rect(img, rectangle))
        return patchfilename


def process_images(img_files, date_folder):
    """Process images and generate thumbnail patches for existing JSON detections."""

    total_img_files = len(img_files)
    patch_folder_path = Path(date_folder + "/patches")
    patch_folder_path.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(img_files):
        image_path = os.path.join(date_folder, filename)
        json_path = os.path.join(date_folder, filename[:-4] + ".json")

        processed_files = idx + 1
        progress = (processed_files / total_img_files) * 100
        print(f"({progress:.2f}%) Processing:  {filename} ")

        if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
            print(f"Skipping {filename}: Image file is missing or empty.")
            continue

        if os.path.isfile(json_path):
            print(json_path)
            print("Json exists for this image file")
            try:
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)
                    print("creating thumbnails for img+json pair")
                    generateThumbnailPatches_JSON(
                        image_path,
                        json_data,
                        patch_folder_path,
                    )
            except json.JSONDecodeError:
                print(f"{filename}: Corrupted JSON file.")


# This code will only run if this script is executed directly
if __name__ == "__main__":
    date_folders = find_date_folders(INPUT_PATH)
    for date_folder_path in date_folders:
        print(date_folders)
        images = scan_for_images(date_folder_path)
        process_images(images, date_folder_path)
