#!/usr/bin/env python3
import os
import json
from pathlib import Path
from PIL import Image
import piexif
import argparse

from core.common import find_date_folders, find_detection_matches, update_main_list

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


def load_anylabeling_data(json_path, image_path):
    """Loads data from an AnyLabeling JSON file.

    Args:

    """

    with open(json_path, "r") as f:
        data = json.load(f)

    long = data["longitude"]
    lat = data["latitude"]

    # Extract relevant data from the detection labels
    detections = data["shapes"]

    nightfolder = os.path.dirname(image_path)
    i = 0
    for label in detections:
        the_patch_path = label["patch_path"]

        full_patch_path = Path(
            nightfolder + "/" + the_patch_path
        )  # should work on mac or windows

        print(str(i) + "/" + str(len(detections)) + " detection being processed")
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
    # Load image
    img = Image.open(input_path)

    # Try to load existing EXIF data, or start fresh
    exif_bytes = img.info.get("exif")
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Check if GPS data already exists
    gps_existing = exif_dict.get("GPS", {})
    # Skipping the Skipping for now!
    # if gps_existing.get(piexif.GPSIFD.GPSLatitude) and gps_existing.get(piexif.GPSIFD.GPSLongitude):
    #    print("GPS data already exists. No changes made.")
    #    return

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

    # Inject GPS into EXIF
    exif_dict["GPS"] = gps_ifd
    exif_bytes = piexif.dump(exif_dict)

    # Save the image with new EXIF
    img.save(output_path, exif=exif_bytes)
    print(f"Saved image with GPS data: {output_path}")


def connect_metadata_matched_img_json_pairs(
    hu_matched_img_json_pairs, bot_matched_img_json_pairs
):

    # Process Human Detections
    print("processing Human Detections.........")
    if ID_HUMANDETECTIONS:
        # Next process each pair and generate temporary files for the ROI of each detection in each image
        # Iterate through image-JSON pairs
        index = 0
        numofpairs = len(hu_matched_img_json_pairs)
        for pair in hu_matched_img_json_pairs:

            # Load JSON file
            image_path, json_path = pair[:2]  # Always extract the first two elements

            load_anylabeling_data(json_path, image_path)

    print("processing BOT Detections.........")
    if ID_BOTDETECTIONS:
        # Next process each pair and generate temporary files for the ROI of each detection in each image
        # Iterate through image-JSON pairs
        index = 0
        numofpairs = len(bot_matched_img_json_pairs)
        for pair in bot_matched_img_json_pairs:
            # Load JSON file and
            image_path, json_path = pair[:2]  # Always extract the first two elements

            load_anylabeling_data(json_path, image_path)


def run(input_path):
    global INPUT_PATH, ID_HUMANDETECTIONS, ID_BOTDETECTIONS
    INPUT_PATH = input_path
    ID_HUMANDETECTIONS = True
    ID_BOTDETECTIONS = True

    print("adding exif info to the patches")

    """
    First the script takes in a INPUT_PATH

    Then, (to simplify its searching) it looks through all the folders for folders that are just a single "night"
    and follow the date format YYYY-MM-DD for their structure

    in each of these folders, it looks to see if there are any .json

    """
    print("Starting script to  add metadata to raw iamges")
    # ~~~~~~~~~~~~~~~~ GATHERING DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Find all the dated folders that our data lives in
    print("Looking in this folder for MothboxData: " + INPUT_PATH)
    date_folders = find_date_folders(INPUT_PATH)
    print(
        "Found ",
        str(len(date_folders)) + " dated folders potentially full of mothbox data",
    )

    # Look in each dated folder for .json detection files and the matching .jpgs
    hu_matched_img_json_pairs = []
    bot_matched_img_json_pairs = []

    for folder in date_folders:
        hu_list_of_matches, bot_list_of_matches = find_detection_matches(folder)
        hu_matched_img_json_pairs = update_main_list(
            hu_matched_img_json_pairs, hu_list_of_matches
        )
        bot_matched_img_json_pairs = update_main_list(
            bot_matched_img_json_pairs, bot_list_of_matches
        )

    print(
        "Found ",
        str(len(hu_matched_img_json_pairs))
        + " pairs of images and HUMAN detection data insert exif",
    )
    # Example Pair
    print("example human detection and json pair:")
    if len(hu_matched_img_json_pairs) > 0:
        print(hu_matched_img_json_pairs[0])

    print(
        "Found ",
        str(len(bot_matched_img_json_pairs))
        + " pairs of images and BOT detection data to insert exif",
    )
    # Example Pair
    print("example human detection and json pair:")
    if len(bot_matched_img_json_pairs) > 0:
        print(bot_matched_img_json_pairs[0])

    # ~~~~~~~~~~~~~~~~ Processing Data ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Now that we have our data to be processed in a big list, process all
    # detections and write non-taxonomic EXIF data (GPS and existing fields).
    connect_metadata_matched_img_json_pairs(
        hu_matched_img_json_pairs,
        bot_matched_img_json_pairs,
    )

    print("Finished Attaching exif info")


if __name__ == "__main__":
    args = parse_args()
    run(args.input_path)
