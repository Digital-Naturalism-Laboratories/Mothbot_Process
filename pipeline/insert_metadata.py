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
    find_images_recursive,
    find_detection_matches,
    find_detection_matches_processed,
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

# ~~~~Flexible Field Sheet Support~~~~~~~
#
# Different scientists customize their field sheets (extra columns like
# "timelapse_interval", multiple attractors, "country" instead of "dataset",
# etc.). Two things make that flexible instead of breaking the script:
#
# 1) Every column that exists in a field sheet's CSV gets copied onto the
#    output JSON automatically (see "field_sheet_metadata" in
#    load_anylabeling_data below), under whatever name the scientist used
#    for it. New/unexpected columns are never dropped.
#
# 2) A handful of "canonical" fields are ALSO copied to fixed keys
#    (data["device"], data["latitude"], etc.) because other scripts in this
#    pipeline (e.g. export_csv.py) expect those exact keys to exist on every
#    sample. FIELD_ALIASES lets a canonical field be filled in from whichever
#    of several possible column names a given sheet actually used. If you
#    see a field sheet that calls something by yet another name, just add it
#    to the list here -- no other code needs to change.
#
# Column name matching is case/whitespace-insensitive and treats spaces the
# same as underscores (e.g. "Deploy Date", "deploy_date", and "DEPLOY_DATE"
# are all treated as the same column).
FIELD_ALIASES = {
    "device": ["device"],
    "device_name": ["device_name"],
    "firmware": ["firmware"],
    "sheet": ["sheet"],
    "dataset": ["dataset"],
    "project": ["project"],
    "site": ["site"],
    "latitude": ["latitude"],
    "longitude": ["longitude"],
    "height_above_ground": ["height_above_ground"],
    "deployment_name": ["deployment_name"],
    "UTC": ["UTC"],
    "deployment_date": ["deployment_date", "deploy_date"],
    "collect_date": ["collect_date"],
    "data_storage_location": ["data_storage_location"],
    "crew": ["crew"],
    "notes": ["notes"],
    "schedule": ["schedule"],
    "habitat": ["habitat"],
    # Sheets supporting multiple attractors use attractor1/2/3 instead of a
    # single "attractor" column. Fall back to attractor1 so the legacy
    # "attractor" field (used by export_csv.py) still gets a sensible value;
    # attractor1/2/3 and their *_settings columns are also preserved in full
    # via field_sheet_metadata regardless of this fallback.
    "attractor": ["attractor", "attractor1"],
    "attractor_location": ["attractor_location"],
}


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
    if not raw_height:
        return None
    # Use regular expression to find the first floating-point or integer number
    match = re.search(r"[-+]?\d+\.?\d*|\d+", str(raw_height))
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


def get_aliased(metadata, canonical_name, default=""):
    """Look up a canonical field's value in a (normalized) metadata row,
    trying each of its known aliases (see FIELD_ALIASES) in order and
    returning the first one that's actually present and non-empty.

    `metadata` is expected to already have normalized (lowercase,
    whitespace/space-vs-underscore insensitive) keys -- see _normalize_row.
    """
    for alias in FIELD_ALIASES.get(canonical_name, [canonical_name]):
        key = _normalize_key(alias)
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return default


def load_anylabeling_data(json_path, image_path, metadata):
    """Writes field-sheet metadata into an AnyLabeling-style detection JSON.

    `metadata` is one normalized row from the field sheet CSV (see
    _normalize_row/find_csv_match) -- a dict that may contain any columns a
    given scientist's sheet happens to have, not just the ones this script
    knows about.

    Two things happen here so the script stays flexible:
      1. A fixed set of "well known" fields get copied onto flat keys
         (data["device"], data["latitude"], ...) using FIELD_ALIASES to
         tolerate sheets that name them slightly differently. Other scripts
         in this pipeline (export_csv.py) depend on these exact keys
         existing, so they're always written, defaulting to "" / "0.00000"
         etc. when a sheet doesn't have that column at all.
      2. EVERY column actually present in the sheet -- known or not (e.g.
         timelapse_interval, attractor2, attractor3, attractor1_settings,
         target_material, nearby_vegetation, country, device_name...) -- is
         also written wholesale into data["field_sheet_metadata"], so
         nothing a scientist adds to their sheet is ever silently dropped,
         even if no other part of the pipeline knows what to do with it yet.
    """
    therawgroundheight = get_aliased(metadata, "height_above_ground", "-1")

    with open(json_path, "r") as f:
        data = json.load(f)

        data["filepath"] = image_path
        data["uploaded"] = metadata.get("uploaded", "")
        data["sd"] = metadata.get("sd_card", "")
        data["device"] = get_aliased(metadata, "device")
        data["firmware"] = str(get_aliased(metadata, "firmware"))
        data["sheet"] = get_aliased(metadata, "sheet")
        data["datasetcollection"] = get_aliased(metadata, "dataset")
        data["project"] = get_aliased(metadata, "project")
        data["site"] = get_aliased(metadata, "site")
        data["longitude"] = get_aliased(metadata, "longitude", "0.00000")
        data["latitude"] = get_aliased(metadata, "latitude", "0.00000")
        data["ground_height"] = extract_number(therawgroundheight)
        data["deployment_name"] = get_aliased(metadata, "deployment_name")
        data["UTC"] = get_aliased(metadata, "UTC", "0")
        data["deployment_date"] = get_aliased(metadata, "deployment_date")
        data["collect_date"] = get_aliased(metadata, "collect_date")
        data["data_storage_location"] = get_aliased(metadata, "data_storage_location")
        data["crew"] = get_aliased(metadata, "crew")
        data["notes"] = get_aliased(metadata, "notes")
        data["schedule"] = get_aliased(metadata, "schedule")
        data["habitat"] = get_aliased(metadata, "habitat")
        data["attractor"] = get_aliased(metadata, "attractor")
        data["attractor_location"] = get_aliased(metadata, "attractor_location")

        # Flexible passthrough: keep every column from the field sheet,
        # whatever it's called, so extra/custom info is never lost.
        data["field_sheet_metadata"] = dict(metadata)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("Metadata written into 'Json' field for." + str(json_path))

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


def _normalize_key(key: str) -> str:
    """Normalize a CSV column header so field sheets with different
    capitalization or spacing for the same field (e.g. 'Deploy Date',
    'deploy_date', 'DEPLOY_DATE') are all treated as the same column.
    """
    if key is None:
        return key
    return key.strip().lower().replace(" ", "_")


def _normalize_row(row: dict) -> dict:
    """Returns a copy of a csv.DictReader row with normalized keys (see
    _normalize_key) and whitespace-stripped string values. This is what
    lets the rest of the script -- and any extra/custom columns a
    scientist's sheet happens to have -- be looked up reliably regardless
    of exactly how the sheet's headers were typed.
    """
    normalized = {}
    for key, value in row.items():
        if key is None:
            # csv.DictReader puts any "extra" unnamed columns under the
            # None key as a list; there's nothing meaningful to normalize.
            continue
        norm_key = _normalize_key(key)
        norm_value = value.strip() if isinstance(value, str) else value
        normalized[norm_key] = norm_value
    return normalized


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
            row = _normalize_row(row)
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
            row = _normalize_row(row)
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


def run(input_path, metadata_path, dataset_root=None):
    """Run the metadata-insertion pipeline programmatically.

    Parameters
    ----------
    input_path : str
        Root folder containing Mothbox data (any sub-folder structure).
    metadata_path : str
        Path to the CSV field-sheet metadata file.
    dataset_root : str | None
        Top-level folder for the _processed output tree.  Defaults to
        *input_path* itself.
    """
    global ID_HUMANDETECTIONS, ID_BOTDETECTIONS, INPUT_PATH

    INPUT_PATH = input_path
    _dataset_root = dataset_root or input_path

    # ~~~~~~~~~~~~~~~~ GATHERING DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Looking in this folder for MothboxData: " + input_path)

    # Use structure-agnostic discovery: finds JSONs in the _processed tree
    hu_matched_img_json_pairs, bot_matched_img_json_pairs = (
        find_detection_matches_processed(_dataset_root)
    )

    print(
        "Found ",
        str(len(hu_matched_img_json_pairs))
        + " pairs of images and HUMAN detection data to insert metadata",
    )
    if len(hu_matched_img_json_pairs) > 0:
        print("example human detection and json pair:")
        print(hu_matched_img_json_pairs[0])

    print(
        "Found ",
        str(len(bot_matched_img_json_pairs))
        + " pairs of images and BOT detection data to insert metadata",
    )
    if len(bot_matched_img_json_pairs) > 0:
        print("example bot detection and json pair:")
        print(bot_matched_img_json_pairs[0])

    metadata = find_csv_match(input_path, metadata_path)

    # ~~~~~~~~~~~~~~~~ Processing Data ~~~~~~~~~~~~~~~~~~~~~~~~~~

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