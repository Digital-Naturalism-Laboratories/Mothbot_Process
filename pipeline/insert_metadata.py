#!/usr/bin/env python3

"""
MOTHBOT_InsertMetadata
This script tries to put field sheet metadata into the json files associated with each raw image

Get list of taxa from just specific region in GBIF
ex:
country = 'PA' #2 letter country code https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 "Panama"==PA
classKey = '216' # just insects i think

Example search in GBIF
https://www.gbif.org/occurrence/taxonomy?country=PA&taxon_key=212


Arguments:
  -h, --help    Show this help message and exit

"""

# import polars as pl
import os
import sys
import json
import argparse
import re
import numpy as np
from PIL import Image
from PIL import ImageFile

from datetime import datetime, timedelta
from collections import defaultdict
import csv

from core.common import (
    find_date_folders,
    find_detection_matches,
    update_main_list,
    current_timestamp,
    get_rotated_rect_raw_coordinates,
)

ImageFile.LOAD_TRUNCATED_IMAGES = (
    True  # makes ok for use images that are messed up slightly
)

# ~~~~Variables to Change~~~~~~~

INPUT_PATH = r"G:\Shared drives\Mothbox Management\Testing\ExampleDataset\Les_BeachPalm_hopeCobo_2025-06-20"  # raw string

METADATA_PATH = r"..\Mothbox_Main_Metadata_Field_Sheet_Example - Form responses 1.csv"
# UTC_OFFSET= 8 # The file shou Panama is -5, Indonesia is 8 change for different locations

TAXA_LIST_PATH = r"..\SpeciesList_CountryIndonesia_TaxaInsecta.csv"  # downloaded from GBIF for example just insects in panama: https://www.gbif.org/occurrence/taxonomy?country=PA&taxon_key=216


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
    parser.add_argument(
        "--metadata",
        required=False,
        default=METADATA_PATH,
        help="path to csv of field metadata",
    )

    return parser.parse_args()


# FUNCTIONS ~~~~~~~~~~~~~


def extract_number(raw_height):
    """
    Extracts the numerical value from a string representing height.

    Args:
      raw_height: The string containing the height information.

    Returns:
      The numerical value of the height as a float, or None if no numerical value
      could be extracted.
    """
    # Use regular expression to find the first floating-point or integer number
    match = re.search(r"[-+]?\d+\.?\d*|\d+", raw_height)
    if match:
        return float(match.group(0))
    else:
        return None


def handle_rotation_annotation(points):
    """Converts an oriented bounding box to a horizontal bounding box.

    Args:
      points: A list of points representing the vertices of the oriented bounding box.

    Returns:
      A tuple containing the top, left, width, and height of the horizontal bounding box.
    """

    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")

    for point in points:
        x, y = point
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    top = min_y
    left = min_x
    width = max_x - min_x
    height = max_y - min_y

    return top, left, width, height


# PUt everything in the JSON


def load_anylabeling_data(json_path, image_path, metadata):
    """Loads data from an AnyLabeling JSON file.

    Args:
      json_path: The path to the JSON file.

    Returns:
      A dictionary containing the loaded data.
    """
    latitude = metadata.get("latitude", "0.00000")
    longitude = metadata.get("longitude", "0.00000")
    therawgroundheight = metadata.get("height_above_ground", "-1")

    with open(json_path, "r") as f:
        data = json.load(f)

        data["filepath"] = image_path
        data["uploaded"] = metadata.get("uploaded", "")
        data["sd"] = metadata.get("sd_card", "")
        data["device"] = metadata.get("device", "")
        data["firmware"] = str(metadata.get("firmware", ""))
        data["sheet"] = metadata.get("sheet", "")
        data["datasetcollection"] = metadata.get("dataset", "")
        data["project"] = metadata.get("project", "")
        data["site"] = metadata.get("site", "")
        data["longitude"] = longitude
        data["latitude"] = latitude
        data["ground_height"] = extract_number(therawgroundheight)
        data["deployment_name"] = metadata.get("deployment_name", "")
        data["UTC"] = metadata.get("UTC", "0")
        data["deployment_date"] = metadata.get("deployment_date", "")
        data["collect_date"] = metadata.get("collect_date", "")
        data["data_storage_location"] = metadata.get("data_storage_location", "")
        data["crew"] = metadata.get("crew", "")
        data["notes"] = metadata.get("notes", "")
        data["schedule"] = metadata.get("schedule", "")
        data["habitat"] = metadata.get("habitat", "")
        data["attractor"] = metadata.get("attractor", "")
        data["attractor_location"] = metadata.get("attractor_location", "")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("✅ Metadata written into 'Json' field for." + str(json_path))

    return


# Maybe this?
def connect_metadata_matched_img_json_pairs(
    hu_matched_img_json_pairs, bot_matched_img_json_pairs, metadata
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

            load_anylabeling_data(json_path, image_path, metadata)

    print("processing BOT Detections.........")
    if ID_BOTDETECTIONS:
        # Next process each pair and generate temporary files for the ROI of each detection in each image
        # Iterate through image-JSON pairs
        index = 0
        numofpairs = len(bot_matched_img_json_pairs)
        for pair in bot_matched_img_json_pairs:
            # Load JSON file and
            image_path, json_path = pair[:2]  # Always extract the first two elements

            load_anylabeling_data(json_path, image_path, metadata)


def _without_first_prefix(name: str) -> str:
    """Return the string with the first underscore-separated prefix removed.
    e.g. 'Indonesia_Les_Wilan...' -> 'Les_Wilan...'. If no underscore, returns original.
    """
    if not name:
        return name
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 else name


def find_csv_match(input_path: str, metadata_path: str) -> dict:
    """
    Finds a row in the CSV where 'deployment_name' matches either the folder name
    or its parent folder name of input_path.
    Tolerates the presence/absence of the first leading prefix on either side.
    Matching is case-insensitive.
    If multiple matches are found, prints a warning and returns only the first one.

    Returns:
        dict: The first matching row as a dict, or {} if no match is found.
    """
    parent_folder = os.path.basename(os.path.dirname(input_path)).strip()
    current_folder = os.path.basename(input_path).strip()

    # alternate versions without first prefix
    alt_parent = _without_first_prefix(parent_folder)
    alt_current = _without_first_prefix(current_folder)

    # store variants in lowercase for case-insensitive matching
    folder_variants = {
        parent_folder.lower(),
        alt_parent.lower(),
        current_folder.lower(),
        alt_current.lower(),
    }

    matches = []
    print(f"scanning for metadata matches... (folder variants: {folder_variants})")

    with open(metadata_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dep_name = (row.get("deployment_name") or "").strip()
            if not dep_name:
                continue

            alt_dep = _without_first_prefix(dep_name)
            dep_variants = {dep_name.lower(), alt_dep.lower()}

            # if any variant intersects, it's a match
            if folder_variants & dep_variants:
                matches.append(row)

    if len(matches) > 1:
        print(
            f"⚠️ Warning: Multiple matches found for '{parent_folder}', using the first one."
        )
    if len(matches) == 1:
        print(f"✅ Matched deployment.name = '{matches[0].get('deployment_name')}'")
    return matches[0] if matches else {}


def find_csv_match_old_onlyparent(input_path: str, metadata_path: str) -> dict:
    """
    Finds a row in the CSV where 'deployment.name' matches the folder name of input_path.
    Tolerates the presence/absence of the first leading prefix on either side.
    Matching is case-insensitive.
    If multiple matches are found, prints a warning and returns only the first one.

    Returns:
        dict: The first matching row as a dict, or {} if no match is found.
    """
    parent_folder = os.path.basename(os.path.dirname(input_path)).strip()
    alt_parent = _without_first_prefix(parent_folder)

    # store variants in lowercase for case-insensitive matching
    folder_variants = {parent_folder.lower(), alt_parent.lower()}

    matches = []
    print(f"scanning for metadata matches... (folder variants: {folder_variants})")

    with open(metadata_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dep_name = (row.get("deployment_name") or "").strip()
            if not dep_name:
                continue

            alt_dep = _without_first_prefix(dep_name)
            dep_variants = {dep_name.lower(), alt_dep.lower()}

            # if any variant intersects, it's a match
            if folder_variants & dep_variants:
                matches.append(row)

    if not matches:
        print(
            f"⚠️ No match found for '{parent_folder}' (or '{alt_parent}') in {metadata_path}"
        )
        return {}

    if len(matches) > 1:
        print(
            f"⚠️ Warning: Multiple matches found for '{parent_folder}', using the first one."
        )

    print(f"✅ Matched deployment.name = '{matches[0].get('deployment_name')}'")
    return matches[0]


def run(input_path, metadata_path):
    """Run the metadata-insertion pipeline programmatically.

    Parameters
    ----------
    input_path : str
        Root folder containing dated Mothbox data sub-folders.
    metadata_path : str
        Path to the CSV field-sheet metadata file.
    """
    global ID_HUMANDETECTIONS, ID_BOTDETECTIONS, INPUT_PATH

    INPUT_PATH = input_path

    # ~~~~~~~~~~~~~~~~ GATHERING DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Find all the dated folders that our data lives in
    print("Looking in this folder for MothboxData: " + input_path)
    date_folders = find_date_folders(input_path)
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
        + " pairs of images and HUMAN detection data to insert metadata",
    )
    # Example Pair
    print("example human detection and json pair:")
    if len(hu_matched_img_json_pairs) > 0:
        print(hu_matched_img_json_pairs[0])

    print(
        "Found ",
        str(len(bot_matched_img_json_pairs))
        + " pairs of images and BOT detection data to insert metadata",
    )
    # Example Pair
    print("example human detection and json pair:")
    if len(bot_matched_img_json_pairs) > 0:
        print(bot_matched_img_json_pairs[0])

    metadata = find_csv_match(input_path, metadata_path)

    # ~~~~~~~~~~~~~~~~ Processing Data ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Now that we have our data to be processed in a big list, it's time to load up the Pybioclip stuff
    connect_metadata_matched_img_json_pairs(
        hu_matched_img_json_pairs,
        bot_matched_img_json_pairs,
        metadata=metadata,
    )

    print("Finished Attaching Metadata field info")


if __name__ == "__main__":

    print("Starting script to  add metadata to raw iamges")
    args = parse_args()
    ID_BOTDETECTIONS = bool(int(args.ID_Bot))
    ID_HUMANDETECTIONS = bool(int(args.ID_Hum))

    run(args.input_path, args.metadata)
