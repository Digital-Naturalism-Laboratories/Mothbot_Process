#!/usr/bin/env python3

"""
MOTHBOT_ID
This script looks for mothbox image data and detection data, pairs them together, finds the region of interest in the image of the detection
feeds this ROI to pyBIOCLIP to try to get an ID

the pybioclip also takes in GBIF species lists


Get list of taxa from just specific region in GBIF
ex:
country = 'PA' #2 letter country code https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 "Panama"==PA
classKey = '216' # just insects i think

Example search in GBIF
https://www.gbif.org/occurrence/taxonomy?country=PA&taxon_key=212


Usage:
  python Mothbox_ID.py

Arguments:
  -h, --help    Show this help message and exit

"""
import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # needed for some macs to automatically download files associated with some of the libraries
import polars as pl
import os
import sys
import json
import argparse
import re
import io
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = (
    True  # makes ok for use images that are messed up slightly
)
from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier
from bioclip.predict import create_classification_dict
import importlib.metadata

VERSION = "pybioclip_" + importlib.metadata.version("pybioclip")

from core.common import (
    find_date_folders,
    find_detection_matches,
    update_main_list,
    current_timestamp,
    get_rotated_rect_raw_coordinates,
    get_device,
    print_device_info,
)

# ~~~~Variables to Change~~~~~~~

INPUT_PATH = r"D:\MothboxData_Hubert\data\Panama\Hoya_119m_bothDeer_2025-01-26\2025-01-26"  # raw string
SPECIES_LIST = r"../SpeciesList_CountryPanamaCostaRica_TaxaInsecta_doi.org10.15468dl.6nxkw6.csv"  # downloaded from GBIF for example just insects in panama: https://www.gbif.org/occurrence/taxonomy?country=PA&taxon_key=212


""" KINGDOM = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6"""

TAXONOMIC_RANK_FILTER_num = 3  #!!! change this number to change the taxonomic rank we filter with. IE filter to order with "3" or filter to genus with "5"

# you can See if a json file has an existing ID by looking at identifier_bot: pybioclip
OVERWRITE_EXISTING_IDs = True  # True

# you probably always want these below as true
ID_HUMANDETECTIONS = True
ID_BOTDETECTIONS = True

# ~~~~Other Global Variables~~~~~~~

TAXA_COLS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
TAXONOMIC_RANK_FILTER = Rank.ORDER
TOL_TAXONOMIC_RANK = "species"  # Change this to "species" to target just the species in your CSV # Note i think this is actually just always needs to be set for SPECIES for this example
DOMAIN = "Eukarya"  # basically our "creature" tag? figure we will never see a prokaryote on the mothbox # Also i think GBIF has a "Biota" category that is a fancier version of "creature" or "life"
taxa_path = SPECIES_LIST

# print(torch.cuda.is_available())

# TODO: Re-enable CUDA once pybioclip batch performance on GPU is fixed.
# Original line: 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOI = ""


from importlib.metadata import version
print(f"pybioclip version: {version('pybioclip')}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=False,
        default=INPUT_PATH,
        help="path to images for classification (ex: datasets/test_images/data)",
    )
    parser.add_argument(
        "--TOLrank",
        default=TOL_TAXONOMIC_RANK,
        # help="rank to which to classify; must be column in --taxa-csv (default: {TAXONOMIC_RANK})", #this always needs to just be left at species i think
    )
    parser.add_argument(
        "--rank",
        default=TAXONOMIC_RANK_FILTER_num,
        help="rank to which to classify; must be column in --taxa-csv (default: {TAXONOMIC_RANK})",
    )
    parser.add_argument(
        "--flag-det-errors",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to flag detection errors like holes and smudges (default: --flag-det-errors)",
    )
    parser.add_argument(
        "--taxa_csv",
        default=SPECIES_LIST,
        help="CSV with taxonomic labels to use for CustomClassifier (default: {SPECIES_LIST})",
    )
    parser.add_argument(
        "--taxa_cols",
        default=TAXA_COLS,
        help=f"taxonomic columns in taxa CSV to load (default: {TAXA_COLS})",
    )
    parser.add_argument(
        "--device",
        required=False,
        choices=["cpu", "cuda"],
        default=DEVICE,
        help="device on which to run pybioblip ('cpu' or 'cuda', default: what your comp detects)",
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
        "--overwrite_prev_bot_ID",
        default=OVERWRITE_EXISTING_IDs,
        required=False,
        help="If IDs already exist, should we overwrite?",
    )

    return parser.parse_args()


# FUNCTIONS ~~~~~~~~~~~~~


def load_taxon_keys(
    taxa_path, taxa_cols=None, taxon_rank="order", flag_det_errors=True
):
    """
    Read taxa_path (path, bytes, or file-like) robustly handling encoding issues
    and return a set of unique, lowercased values for taxon_rank.

    Returns:
        set(str): lowercased unique taxon_rank values
    """
    print(f"Reading {taxa_path!s}, extracting {taxon_rank} values.")

    # encodings to try in order (utf-8 first, then common windows/latin fallbacks)
    encodings = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")

    raw = None
    text = None

    # Accept bytes/bytearray directly
    if isinstance(taxa_path, (bytes, bytearray)):
        raw = bytes(taxa_path)

    # Accept an already-open file-like object
    elif hasattr(taxa_path, "read"):
        try:
            # try reading bytes first (some file-like objects are in binary mode)
            raw = taxa_path.read()
        except Exception:
            # if that fails, try text-mode read
            taxa_path.seek(0)
            text = taxa_path.read()

    # Otherwise treat it as a filesystem path-like
    else:
        p = Path(taxa_path)
        if not p.exists():
            raise FileNotFoundError(f"taxa_path not found: {taxa_path}")
        with open(p, "rb") as f:
            raw = f.read()

    # If we already have text (not bytes), use it
    if text is None:
        # If raw is already str (unlikely) use it
        if isinstance(raw, str):
            text = raw
        else:
            # Try the encodings in order
            decoded = None
            for enc in encodings:
                try:
                    decoded = raw.decode(enc)
                    # quick sanity check: if decoding produced something non-empty, accept it
                    if decoded is not None:
                        text = decoded
                        break
                except Exception:
                    continue
            # Last-resort: decode with replacement so we never raise UnicodeDecodeError
            if text is None:
                text = raw.decode("utf-8", errors="replace")

    # Try to load into Polars; prefer tab-separated but fall back to automatic parsing.
    try:
        df = pl.read_csv(io.StringIO(text), separator="\t")
    except Exception:
        try:
            df = pl.read_csv(io.StringIO(text))  # let polars infer delimiter
        except Exception:
            # final fallback: use pandas to parse then convert to polars
            import pandas as pd

            df_pd = pd.read_csv(io.StringIO(text), sep="\t")
            df = pl.from_pandas(df_pd)

    # If user provided a taxa_cols mapping/dictionary, prefer it for column lookup
    chosen_col = None
    if taxa_cols and isinstance(taxa_cols, dict):
        chosen_col = taxa_cols.get(taxon_rank, None)

    # If chosen_col not set or not present, do case-insensitive matching against dataframe columns
    if not chosen_col or chosen_col not in df.columns:
        cols_lower = {c.lower(): c for c in df.columns}
        if taxon_rank.lower() in cols_lower:
            chosen_col = cols_lower[taxon_rank.lower()]
        elif chosen_col and chosen_col in df.columns:
            # keep chosen_col if it exists
            pass
        else:
            raise KeyError(
                f"taxon_rank '{taxon_rank}' not found in file columns: {list(df.columns)}"
            )

    # Extract unique non-null values and normalize to lowercase strings
    try:
        vals = df[chosen_col].drop_nulls().unique().to_list()
        target_values = {
            str(v).lower() for v in vals if v is not None and str(v).strip() != ""
        }
    except Exception:
        # conservative fallback: iterate rows if the vectorized route fails
        vals = []
        for rec in df.select(chosen_col).to_dicts():
            v = rec.get(chosen_col)
            if v is not None:
                vals.append(v)
        target_values = {str(v).lower() for v in vals if str(v).strip() != ""}

    print("Found", len(target_values), taxon_rank, "values.")
    return target_values


def calculate_rotation_angle(points):
    """Calculates the rotation angle of the rectangle."""
    p1, p2 = points[:2]
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
    return angle


def extract_rectangle_coordinates(points):
    """Extracts rectangle coordinates from a list of points."""
    # Assuming points are in clockwise order (adjust if needed)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    width = max_x - min_x
    height = max_y - min_y
    angle = calculate_rotation_angle(points)
    return min_x, min_y, width, height, angle


def crop_image(image, x, y, w, h):
    """Crops an image based on the specified coordinates."""
    cropped_image = image.crop((x, y, x + w, y + h))
    return cropped_image


def get_bioclip_predictions_batch(imgs, classifier, batch_size=32):
    """Process a batch of PIL images using pybioclip's batch API."""
    results = []
    total = len(imgs)
    total_batches = (total + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_num, i in enumerate(range(0, total, batch_size)):
        batch = imgs[i:i + batch_size]
        img_embeddings = classifier.create_image_features(batch)
        for probs in classifier.create_probabilities(img_embeddings, classifier.txt_embeddings):
            winner = ""
            winnerprob = ""
            winningdict = {}
            for index, pred in enumerate(classifier.format_grouped_probs(
                "", probs, rank=TAXONOMIC_RANK_FILTER, min_prob=1e-9, k=1
            )):
                if index == 0:
                    winner = pred[str(TAXONOMIC_RANK_FILTER.get_label())]
                    winnerprob = pred["score"]
                    winningdict = pred
                    break
            results.append((winner, winnerprob, winningdict))

        # Progress update after each batch
        batches_done = batch_num + 1
        images_done = min(i + batch_size, total)
        elapsed = time.time() - start_time

        if batches_done == 1 and total_batches > 1:
            eta_seconds = (elapsed / batches_done) * (total_batches - batches_done)
            print(f"   ⏱️ First batch done in {elapsed:.1f}s — estimated {eta_seconds:.0f}s remaining ({eta_seconds/60:.1f} min)")
        elif batches_done % 5 == 0 or batches_done == total_batches:
            eta_seconds = (elapsed / batches_done) * (total_batches - batches_done)
            print(f"   📦 Batch {batches_done}/{total_batches} — {images_done}/{total} images — ~{eta_seconds:.0f}s remaining")

    total_time = time.time() - start_time
    print(f" Batch predictions complete — {total} images in {total_time:.1f}s ({total_time/60:.1f} min)")
    return results


def read_cluster_id(json_path, shape_idx):
    """Read clusterID from a shape in a JSON file. Returns None if not present."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if 0 <= shape_idx < len(data["shapes"]):
            cluster_val = data["shapes"][shape_idx].get("clusterID", None)
            if cluster_val is not None:
                return float(cluster_val)
    except Exception:
        pass
    return None


def apply_id_to_cluster(json_paths, idxes, pred, conf, winningdict):
    """Write the same ID result to every member of a cluster."""
    for json_path, idx in zip(json_paths, idxes):
        update_json_labels_and_scores(json_path, idx, pred, conf, winningdict)


def update_json_labels_and_scores(json_path, index, pred, conf, winningdict):
    """Updates the label and score entries for a specific shape in a JSON file.

    Args:
        json_path: The path to the JSON file.
        index: The index of the shape to update (0-based).
        pred: The new label value.
        conf: The new score value.
        winningdict: Full prediction dict containing taxonomic rank values.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if 0 <= index < len(data["shapes"]):
        shape = data["shapes"][index]

        shape["identifier_bot"] = VERSION
        shape["species_list"] = DOI
        shape["timestamp_ID_bot"] = current_timestamp()
        shape["confidence_ID"] = conf

        predstring = str(pred).strip().lower()
        if predstring in ["hole", "background", "wall", "floor", "blank", "sky"]:
            shape["label"] = "ERROR_" + pred
        else:
            shape["label"] = (
                str(TAXONOMIC_RANK_FILTER).replace("Rank.", "") + "_" + pred
            )

        # Add taxonomic ranks only if they exist in the winningdict
        for rank in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
            if rank in winningdict:
                if winningdict[rank].strip().lower() in ["hole", "background", "wall", "floor", "blank", "sky"]:
                    shape[rank] = "ERROR_" + winningdict[rank]
                else:
                    shape[rank] = winningdict[rank]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def add_metadata_to_json(json_path, metadata_path):
    """Adds metadata from a separate JSON file to an existing JSON file.

    Args:
      json_path: The path to the JSON file to modify.
      metadata_path: The path to the JSON file containing the metadata to add.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Check if the 'metadata' key exists in the data
    if "metadata" not in data:
        data["metadata"] = []  # Create an empty 'metadata' list if it doesn't exist

    # Add metadata to the existing 'metadata' list, avoiding duplicates
    for key, value in metadata.items():
        if not any(item.get(key) for item in data["metadata"]):
            data["metadata"].append({key: value})

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Metadata added to {json_path}")


# Patch get_txt_names to always use UTF-8
def fixed_get_txt_names(self):
    txt_names_json = self.get_cached_datafile("embeddings/txt_emb_species.json")
    with open(txt_names_json, encoding="utf-8") as fd:
        return json.load(fd)


def build_classifier(taxa_path, taxa_cols, taxon_rank, device, flag_the_det_errors):
    """Build (or load from cache) a TreeOfLifeClassifier filtered to the given taxa.

    The filtered text embeddings are cached as a .pt file alongside the CSV so
    subsequent runs skip the expensive rebuild.

    Args:
        taxa_path: Path to the GBIF species-list CSV.
        taxa_cols: Column name list for the CSV.
        taxon_rank: Taxonomic rank string (e.g. "order", "species").
        device: Torch device string ("cpu" or "cuda").
        flag_the_det_errors: Whether to add abiotic error labels.

    Returns:
        TreeOfLifeClassifier with txt_names and txt_embeddings filtered to taxa.
    """
    cache_path = os.path.splitext(taxa_path)[0] + ".pt"

    TreeOfLifeClassifier.get_txt_names = fixed_get_txt_names  # UTF-8 patch

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location=device)
        classifier = TreeOfLifeClassifier(device=device)
        classifier.txt_names = cache["txt_names"]
        classifier.txt_embeddings = cache["txt_embeddings"].to(device)
        print("TOL: Loaded number of labels:", len(classifier.txt_names))
        print("TOL: Loaded embeddings shape:", classifier.txt_embeddings.shape)
        return classifier

    # ── No cache → build fresh ──────────────────────────────────────
    taxon_keys_list = load_taxon_keys(
        taxa_path=taxa_path,
        taxa_cols=taxa_cols,
        taxon_rank=taxon_rank.lower(),
        flag_det_errors=flag_the_det_errors,
    )

    print("Loading TOL classifier")
    classifier = TreeOfLifeClassifier(device=device)
    print("TOL: number of labels:", len(classifier.txt_names))
    print("TOL: embeddings shape:", classifier.txt_embeddings.shape)

    print("Finding embeddings matching the targets.")
    found_items = [
        (i, txt_name)
        for i, txt_name in enumerate(classifier.txt_names)
        if create_classification_dict(txt_name, Rank.SPECIES)[taxon_rank].lower() in taxon_keys_list
    ]
    print(f"Found {len(found_items)} embeddings matching the {taxon_rank} values")

    print("Building the filtered embedding tensor")
    txt_feature_ary = [classifier.txt_embeddings[:, i] for i, _ in found_items]
    new_txt_names = [txt_name for _, txt_name in found_items]

    # Append abiotic / error labels
    custom_labels = ["hole", "background", "wall", "floor", "blank", "sky"]
    clc = CustomLabelsClassifier(custom_labels, device=device)
    for i, label in enumerate(custom_labels):
        txt_feature_ary.append(clc.txt_embeddings[:, i])
        new_txt_names.append([[label, label, label, label, label, "", label], label])

    classifier.txt_names = new_txt_names
    classifier.txt_embeddings = torch.stack(txt_feature_ary, dim=1)
    print("TOL: Updated number of labels:", len(classifier.txt_names))
    print("TOL: Updated embeddings shape:", classifier.txt_embeddings.shape)

    print(f"Saving embeddings cache to {cache_path}")
    torch.save(
        {
            "txt_names": classifier.txt_names,
            "txt_embeddings": classifier.txt_embeddings.cpu(),
        },
        cache_path,
    )

    return classifier


def collect_patches_for_detection_set(matched_img_json_pairs, label):
    """Walk a list of (image_path, json_path) pairs and collect all detection patches.

    Args:
        matched_img_json_pairs: List of (image_path, json_path) tuples.
        label: Human-readable label for progress printing ("HU" or "BOT").

    Returns:
        List of (patchfullpath, json_path, idx, cluster_id) tuples.
    """
    all_patches = []
    numofpairs = len(matched_img_json_pairs)

    for index, pair in enumerate(matched_img_json_pairs, start=1):
        image_path, json_path = pair[:2]
        coordinates_of_detections_list, was_pre_ided_list, thepatch_list = (
            get_rotated_rect_raw_coordinates(json_path)
        )
        print(f"{index}/{numofpairs} | {len(coordinates_of_detections_list)} {label} detections in {json_path}")

        for idx, coordinates in enumerate(coordinates_of_detections_list):
            if was_pre_ided_list[idx] and not OVERWRITE_EXISTING_IDs:
                continue
            patchfullpath = os.path.dirname(image_path) + "/" + thepatch_list[idx]
            cluster_id = read_cluster_id(json_path, idx)
            all_patches.append((patchfullpath, json_path, idx, cluster_id))

    return all_patches


def group_patches_by_cluster(all_patches):
    """Group patches by cluster ID, treating noise/unclustered as individual items.

    Args:
        all_patches: List of (patchfullpath, json_path, idx, cluster_id) tuples.

    Returns:
        cluster_groups: dict mapping cluster key → list of (patchfullpath, json_path, idx).
        clustered_count: number of patches that belong to a named cluster.
        individual_count: number of patches treated individually (noise or no cluster).
    """
    cluster_groups = defaultdict(list)
    noise_counter = 0

    for patchfullpath, json_path, idx, cluster_id in all_patches:
        if cluster_id is None or cluster_id == -1.0:
            unique_key = f"__individual_{noise_counter}__"
            noise_counter += 1
            cluster_groups[unique_key].append((patchfullpath, json_path, idx))
        else:
            # Group by the integer part only — 3.1 and 3.4 → group 3
            perceptual_group = int(cluster_id)
            cluster_groups[perceptual_group].append((patchfullpath, json_path, idx))

    individual_count = sum(
        1 for k in cluster_groups if str(k).startswith("__individual_")
    )
    clustered_count = sum(
        len(v) for k, v in cluster_groups.items()
        if not str(k).startswith("__individual_")
    )
    return cluster_groups, clustered_count, individual_count


def run_id_on_detection_set(matched_img_json_pairs, classifier, label):
    """Collect, cluster-deduplicate, batch-predict, and write IDs for one detection set.

    Args:
        matched_img_json_pairs: List of (image_path, json_path) pairs.
        classifier: Loaded TreeOfLifeClassifier.
        label: "HU" or "BOT" — used only for progress messages.
    """
    print(f"\nProcessing {label} detections...")

    all_patches = collect_patches_for_detection_set(matched_img_json_pairs, label)
    if not all_patches:
        print(f"  No {label} detections to process.")
        return

    cluster_groups, clustered_count, individual_count = group_patches_by_cluster(all_patches)

    representatives = []   # one per cluster — the image we actually run inference on
    cluster_members = []   # all members of that cluster (receives the same result)
    for members in cluster_groups.values():
        representatives.append(members[0])
        cluster_members.append(members)

    num_clusters = len(cluster_groups) - individual_count
    print(
        f"  {label} detections: {len(all_patches)} total — "
        f"{clustered_count} in clusters → {num_clusters} representative IDs needed, "
        f"{individual_count} individual"
    )
    print(f"  Running bioclip on {len(representatives)} representative images (down from {len(all_patches)})...")

    batch_size = 32 if DEVICE=="cuda" else 8
    start_time = time.time()

    # Load representative images, skipping any that can't be opened
    rep_imgs, valid_reps, valid_members = [], [], []
    for rep, members in zip(representatives, cluster_members):
        patchfullpath, json_path, idx = rep
        try:
            rep_imgs.append(Image.open(patchfullpath))
            valid_reps.append(rep)
            valid_members.append(members)
        except Exception as e:
            print(f"  Could not open representative {patchfullpath}: {e}")

    # Batch predict on representatives only
    predictions = get_bioclip_predictions_batch(rep_imgs, classifier, batch_size=batch_size)

    # Write results — apply each prediction to ALL members of that cluster
    for (rep_path, rep_json, rep_idx), members, (pred, conf, winningdict) in zip(
        valid_reps, valid_members, predictions
    ):
        print(f" representative: {os.path.basename(rep_path)}: {pred} ({conf:.3f}) → applied to {len(members)} detection(s) in cluster")
        apply_id_to_cluster(
            [m[1] for m in members],
            [m[2] for m in members],
            pred, conf, winningdict,
        )

    total_time = time.time() - start_time
    print(
        f"✅ {label} ID complete — {len(all_patches)} detections identified "
        f"in {total_time:.1f}s ({total_time/60:.1f} min)"
    )


def ID_matched_img_json_pairs(
    hu_matched_img_json_pairs,
    bot_matched_img_json_pairs,
    taxa_path,
    taxa_cols,
    taxon_rank,
    device,
    flag_the_det_errors,
):
    """Build the classifier once, then ID human and bot detections."""

    classifier = build_classifier(taxa_path, taxa_cols, taxon_rank, device, flag_the_det_errors)

    if ID_HUMANDETECTIONS:
        run_id_on_detection_set(hu_matched_img_json_pairs, classifier, "HU")

    if ID_BOTDETECTIONS:
        run_id_on_detection_set(bot_matched_img_json_pairs, classifier, "BOT")


def extract_doi_from_csv_path(csv_path: str) -> str:
    """
    Extracts the DOI from a filename like:
      SpeciesList_..._doi.org10.15468dl.epzeza.csv
    and returns the full DOI URL.

    Works for any DOI variant formatted as "doi.org10....".
    Returns "no_doi" if no valid DOI is found.
    """
    filename = os.path.basename(csv_path)

    # Try to find the DOI chunk (everything after 'doi.org' up to .csv)
    match = re.search(r"(doi\.org[0-9A-Za-z\.\-]+)", filename)
    if not match:
        return "no_doi"

    doi_raw = match.group(1)  # e.g. "doi.org10.15468dl.epzeza"
    doi_core = doi_raw.replace("doi.org", "")  # e.g. "10.15468dl.epzeza"

    # General DOI rule: starts with "10." and has a slash somewhere
    # Fix by inserting a slash between the prefix and the rest
    m = re.match(r"(10\.\d+)(.+)", doi_core)
    if not m:
        return "no_doi"

    prefix, suffix = m.groups()
    doi_fixed = f"{prefix}/{suffix}"

    return f"https://doi.org/{doi_fixed}"


def run(
    input_path, taxa_csv, rank=3, ID_Hum=True, ID_Bot=True, overwrite_prev_bot_ID=True
):
    """Run the full ID pipeline programmatically.

    Parameters
    ----------
    input_path : str
        Root folder containing mothbox data (date-folders inside).
    taxa_csv : str
        Path to the GBIF species-list CSV.
    rank : int
        Taxonomic rank number (3=order, 5=genus, 6=species, …).
    ID_Hum : bool
        Whether to ID human detections.
    ID_Bot : bool
        Whether to ID bot detections.
    overwrite_prev_bot_ID : bool
        Whether to overwrite existing bot IDs.
    """
    global TAXONOMIC_RANK_FILTER, OVERWRITE_EXISTING_IDs, ID_HUMANDETECTIONS
    global ID_BOTDETECTIONS, INPUT_PATH, DOI, DEVICE

    TAXONOMIC_RANK_FILTER = Rank(int(rank))
    OVERWRITE_EXISTING_IDs = bool(overwrite_prev_bot_ID)
    ID_HUMANDETECTIONS = bool(ID_Hum)
    ID_BOTDETECTIONS = bool(ID_Bot)
    INPUT_PATH = input_path

    DOI = extract_doi_from_csv_path(taxa_csv)
    print("using species list: " + DOI)

    DEVICE = get_device()

    # TODO: Re-enable once pybioclip CUDA performance is fixed.
    print("Note: CUDA temporarily disabled for ID while we figure out what's going on with bioclip and CUDA")
    DEVICE = "cpu"

    print_device_info(selected_device=DEVICE)

    # Find all the dated folders that our data lives in
    print("Looking in this folder for MothboxData: " + INPUT_PATH)
    date_folders = find_date_folders(INPUT_PATH)
    print(f"Found {len(date_folders)} dated folders potentially full of mothbox data")

    # Look in each dated folder for .json detection files and the matching .jpgs
    hu_matched_img_json_pairs = []
    bot_matched_img_json_pairs = []

    for folder in date_folders:
        hu_list_of_matches, bot_list_of_matches = find_detection_matches(folder)
        hu_matched_img_json_pairs = update_main_list(hu_matched_img_json_pairs, hu_list_of_matches)
        bot_matched_img_json_pairs = update_main_list(bot_matched_img_json_pairs, bot_list_of_matches)

    print(f"Found {len(hu_matched_img_json_pairs)} pairs of images and HUMAN detection data to try to ID")
    if hu_matched_img_json_pairs:
        print("example human detection and json pair:", hu_matched_img_json_pairs[0])

    print(f"Found {len(bot_matched_img_json_pairs)} pairs of images and BOT detection data to try to ID")
    if bot_matched_img_json_pairs:
        print("example bot detection and json pair:", bot_matched_img_json_pairs[0])

    ID_matched_img_json_pairs(
        hu_matched_img_json_pairs,
        bot_matched_img_json_pairs,
        taxon_rank=TOL_TAXONOMIC_RANK,
        flag_the_det_errors=True,
        taxa_path=taxa_csv,
        taxa_cols=TAXA_COLS,
        device=DEVICE,
    )

    print("Finished Automatic Identification")


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input_path,
        taxa_csv=args.taxa_csv,
        rank=int(args.rank),
        ID_Hum=bool(int(args.ID_Hum)),
        ID_Bot=bool(int(args.ID_Bot)),
        overwrite_prev_bot_ID=bool(int(args.overwrite_prev_bot_ID)),
    )