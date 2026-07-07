#!/usr/bin/env python3
import os
import json
from pathlib import Path
from PIL import Image
import piexif
import argparse
import re
from core.common import find_detection_matches_processed, update_main_list
from core.paths import resolve_patch_path

# TODO: make work for entire deployment
INPUT_PATH = r"G:\Shared drives\Mothbox Management\Testing\ExampleDataset\Les_BeachPalm_hopeCobo_2025-06-20\2025-06-21"

# you probably always want these below as true
ID_HUMANDETECTIONS = True
ID_BOTDETECTIONS = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=False,
        default=INPUT_PATH,
        help="path to images for classification (ex: datasets/test_images/data)",
    )
    parser.add_argument(
        "--ID_Hum",
        required=False,
        default=ID_HUMANDETECTIONS,
        help="ID detections made by humans?",
    )
    parser.add_argument(
        "--ID_Bot",
        required=False,
        default=ID_BOTDETECTIONS,
        help="ID detections made by robots?",
    )

    return parser.parse_args()


def find_image_json_pairs(input_dir):
    """Finds pairs of image and JSON files with the same name in a given directory.

    Args:
      input_dir: The directory to search for files.

    Returns:
      A list of tuples, where each tuple contains the paths to the image and JSON files.
    """

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ]
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]

    pairs = []
    for image_file in image_files:
        json_file = image_file[:-4] + ".json"
        if json_file in json_files:
            pairs.append(
                (
                    os.path.join(input_dir, image_file),
                    os.path.join(input_dir, json_file),
                )
            )

    return pairs


def load_anylabeling_data(json_path, image_path, dataset_root):
    """Loads data from an AnyLabeling JSON file and writes GPS EXIF into each patch."""

    with open(json_path, "r") as f:
        data = json.load(f)

    long = data["longitude"]
    lat = data["latitude"]

    detections = data["shapes"]

    i = 0
    for label in detections:
        the_patch_path = label["patch_path"]

        full_patch_path = Path(
            resolve_patch_path(the_patch_path, image_path, dataset_root)
        )

        print(str(i + 1) + "/" + str(len(detections)) + " detection being processed")
        print("adding GPS to " + str(full_patch_path))
        add_gps_exif(full_patch_path, full_patch_path, float(lat), float(long))

        print("exif data written into patch file" + str(full_patch_path))

        i = i + 1

    return


def generate_patch_dataset(
    dataset, output_dir=INPUT_PATH + "/patches", target_size=(1024, -1)
):
    """
    Generates thumbnails for images in a FiftyOne dataset, skipping existing ones.

    Args:
        dataset: The FiftyOne dataset.
        output_dir: The directory to save the thumbnails.
        target_size: The target size for the thumbnails (width, height).

    Returns:
        None
    """
    patch_folder_path = Path(INPUT_PATH + "/patches")
    patch_folder_path.mkdir(parents=True, exist_ok=True)

    samples_to_process = []
    patch_samples = []

    for sample in dataset.iter_samples(progress=True):
        # filename = os.path.basename(sample.filepath) #this is just the basename that it stores!
        # sample_fullpath=INPUT_PATH+"/"+filename

        # print(sample.filename)

        # print(sample)
        detections = sample.creature_detections.detections
        detnum = 0

        for detection in detections:
            patchfullpath = INPUT_PATH + "/" + detection.patch_path
            # inferred_patchfilename=filename.split('.')[0] + "_" + str(detnum) +"_"+detector+ "." +filename.split('.')[1]
            # inferred_patchfullpath = Path(patch_folder_path) / f'{inferred_patchfilename}'

            # add GPS info to the thumbnail patch
            print("adding GPS to " + patchfullpath)
            add_gps_exif(
                patchfullpath,
                patchfullpath,
                float(sample.latitude),
                float(sample.longitude),
            )

            detnum = detnum + 1

        # sample.save()

    patch_ds = fo.Dataset()
    patch_ds.add_samples(patch_samples)

    patch_ds.app_config["media_fields"] = ["filepath", "filepath_fullimage"]
    patch_ds.app_config["grid_media_field"] = "filepath"
    patch_ds.app_config["modal_media_field"] = "filepath"
    patch_ds.save()

    dataset.save()
    return patch_ds


def deg_to_dms_rational(deg_float):
    """Convert decimal degrees to degrees, minutes, seconds in rational format"""
    deg = int(deg_float)
    min_float = abs(deg_float - deg) * 60
    minute = int(min_float)
    sec_float = (min_float - minute) * 60
    sec = int(sec_float * 10000)

    return ((abs(deg), 1), (minute, 1), (sec, 10000))

def add_gps_exif(input_path, output_path, lat, lng, altitude=None):
    img = Image.open(input_path)

    exif_bytes = img.info.get("exif")
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Extract datetime from filename — supports both formats:
    #   ISO 8601:  name_2026-07-07T03-39-06+02-00_...  (colons replaced with dashes for filename safety)
    #   Legacy:    name_2026_07_07__03_39_06_...
    filename = Path(input_path).name
    match = re.search(r"_(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})", filename)
    if not match:
        match = re.search(r"_(\d{4})_(\d{2})_(\d{2})__?(\d{2})_(\d{2})_(\d{2})", filename)
    if match:
        y, m, d, h, mi, s = match.groups()
        datetime_str = f"{y}:{m}:{d} {h}:{mi}:{s}".encode("utf-8")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = datetime_str
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = datetime_str
        exif_dict["0th"][piexif.ImageIFD.DateTime] = datetime_str
    else:
        print(f"No datetime found in filename: {filename}")

    # Create GPS IFD
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lng >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(lng),
    }

    if altitude is not None:
        gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0 if altitude >= 0 else 1
        gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(abs(altitude * 100)), 100)

    exif_dict["GPS"] = gps_ifd
    exif_bytes = piexif.dump(exif_dict)

    img.save(output_path, exif=exif_bytes)
    print(f"Saved image with GPS and datetime data: {output_path}")

def connect_metadata_matched_img_json_pairs(
    hu_matched_img_json_pairs, bot_matched_img_json_pairs, dataset_root
):

    # Process Human Detections
    print("processing Human Detections.........")
    if ID_HUMANDETECTIONS:
        for pair in hu_matched_img_json_pairs:
            image_path, json_path = pair[:2]
            load_anylabeling_data(json_path, image_path, dataset_root)

    print("processing BOT Detections.........")
    if ID_BOTDETECTIONS:
        for pair in bot_matched_img_json_pairs:
            image_path, json_path = pair[:2]
            load_anylabeling_data(json_path, image_path, dataset_root)


def run(input_path, dataset_root=None):
    global INPUT_PATH, ID_HUMANDETECTIONS, ID_BOTDETECTIONS
    INPUT_PATH = input_path
    ID_HUMANDETECTIONS = True
    ID_BOTDETECTIONS = True

    _dataset_root = dataset_root or input_path

    print("adding exif info to the patches")
    print("Looking in this folder for MothboxData: " + INPUT_PATH)

    # ~~~~~~~~~~~~~~~~ GATHERING DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

    hu_matched_img_json_pairs, bot_matched_img_json_pairs = (
        find_detection_matches_processed(_dataset_root, source_folder=input_path)
    )

    print(
        "Found ",
        str(len(hu_matched_img_json_pairs))
        + " pairs of images and HUMAN detection data to insert exif",
    )
    if len(hu_matched_img_json_pairs) > 0:
        print("example human detection and json pair:")
        print(hu_matched_img_json_pairs[0])

    print(
        "Found ",
        str(len(bot_matched_img_json_pairs))
        + " pairs of images and BOT detection data to insert exif",
    )
    if len(bot_matched_img_json_pairs) > 0:
        print("example bot detection and json pair:")
        print(bot_matched_img_json_pairs[0])

    # ~~~~~~~~~~~~~~~~ Processing Data ~~~~~~~~~~~~~~~~~~~~~~~~~~

    connect_metadata_matched_img_json_pairs(
        hu_matched_img_json_pairs,
        bot_matched_img_json_pairs,
        dataset_root=_dataset_root,
    )

    print("Finished Attaching exif info")


if __name__ == "__main__":
    args = parse_args()
    run(args.input_path)
