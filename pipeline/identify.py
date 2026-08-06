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
import hashlib
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
import pandas as pd
from bioclip import TreeOfLifeClassifier, Rank
import importlib.metadata

VERSION = "pybioclip_" + importlib.metadata.version("pybioclip")

from core.common import (
    find_date_folders,
    find_detection_matches,
    find_detection_matches_processed,
    update_main_list,
    current_timestamp,
    get_rotated_rect_raw_coordinates,
    get_device,
    print_device_info,
)
from core.paths import resolve_patch_path

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

DATASET_ROOT = None  # Set by run(); when set, patch paths are resolved via _processed tree
TAXA_COLS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
TAXONOMIC_RANK_FILTER = Rank.ORDER
TOL_TAXONOMIC_RANK = "species"  # Change this to "species" to target just the species in your CSV # Note i think this is actually just always needs to be set for SPECIES for this example
DOMAIN = "Eukarya"  # basically our "creature" tag? figure we will never see a prokaryote on the mothbox # Also i think GBIF has a "Biota" category that is a fancier version of "creature" or "life"
taxa_path = SPECIES_LIST

# print(torch.cuda.is_available())

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
    if not taxa_path or (isinstance(taxa_path, str) and not taxa_path.strip()):
        print("No species list provided — running without taxon filter (full Tree of Life).")
        return set()

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


# Representatives are ID'd and written to disk in chunks of this size. Smaller =
# finer progress feedback + more frequent crash-safe banking, but slightly more
# per-call predict() overhead. The main pipeline chunks explicitly (see
# run_id_on_detection_set); the whole point is that predict() is NOT called once
# on the entire dataset — that is what made the final aggregation opaque and
# unbanked. See docstring in run_id_on_detection_set for the reasoning.
PREDICT_CHUNK = 256


def get_bioclip_predictions_batch(imgs, classifier, batch_size=32):
    """Predict taxonomic IDs for a list of PIL images using the modern predict() API.

    Kept for single-image / utility use (see get_bioclip_prediction_PILimg). The
    main pipeline no longer routes through here — it chunks predict() calls itself
    so it can bank each chunk to disk and report live progress.

    Returns:
        List of (winner, winnerprob, winningdict) tuples, one per input image.
    """
    rank_label = str(TAXONOMIC_RANK_FILTER.get_label())
    raw_predictions = classifier.predict(
        imgs, rank=TAXONOMIC_RANK_FILTER, k=1, batch_size=batch_size,
    )
    results = []
    for pred in list(raw_predictions):
        winner = pred.get(rank_label, "")
        winnerprob = pred.get("score", 0.0)
        results.append((winner, winnerprob, pred))
    return results


def get_bioclip_prediction_PILimg(img, classifier):
    """Run inference on a single PIL image. Returns (winner, winnerprob, winningdict)."""
    winner, winnerprob, winningdict = get_bioclip_predictions_batch([img], classifier, batch_size=1)[0]
    print(f"  This is the winner: {winner} with a score of {winnerprob}")
    return winner, winnerprob, winningdict


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

        # Archive previous bot ID if it was from a different model or species list.
        old_identifier = shape.get("identifier_bot", "")
        old_doi = shape.get("species_list", "")
        if old_identifier and (old_identifier != VERSION or old_doi != DOI):
            BOT_ID_FIELDS = ["identifier_bot", "species_list", "timestamp_ID_bot",
                             "confidence_ID", "label",
                             "kingdom", "phylum", "class", "order", "family", "genus", "species"]
            snapshot = {k: shape[k] for k in BOT_ID_FIELDS if k in shape}
            shape.setdefault("bot_id_history", []).append(snapshot)

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


def _build_classifier_from_embedding_cache(txt_embeddings, txt_names, device):
    """Create a TreeOfLifeClassifier from pre-filtered embeddings without loading the full TOL set.

    TreeOfLifeClassifier.__init__() always loads the complete Tree of Life
    embedding file (~900 MB numpy array + large JSON).  On a cache hit we can
    skip that entirely: initialise only the vision model via BaseClassifier,
    then inject the small filtered tensors that were saved to cache.

    The resulting object is a fully functional TreeOfLifeClassifier — predict()
    calls get_txt_embeddings() / get_current_txt_names() which both return the
    subset attrs when they are set, so inference is identical to the full path.
    """
    from bioclip.predict import BaseClassifier as _BaseClassifier

    # Allocate the instance without calling TreeOfLifeClassifier.__init__
    # (which would trigger the 900 MB TOL embedding load).
    classifier = object.__new__(TreeOfLifeClassifier)
    # BaseClassifier.__init__ loads the CLIP vision model + torch.compile — unavoidable.
    _BaseClassifier.__init__(classifier, device=device)

    # Inject the pre-filtered subset; full attrs are not needed in this mode.
    classifier.txt_embeddings = None
    classifier.txt_names = None
    classifier._subset_txt_embeddings = txt_embeddings.to(device)
    classifier._subset_txt_names = txt_names
    return classifier


def build_classifier(taxa_path, taxa_cols, taxon_rank, device, flag_the_det_errors):
    """Build (or load from cache) a TreeOfLifeClassifier filtered to the given taxa.

    Cache format (v2): stores the pre-filtered embedding tensor + names list so
    subsequent runs skip loading the full ~900 MB Tree of Life embedding file.

    Old format (v1 bool mask) is detected and upgraded automatically on the next
    full run — delete the .pt file to force a rebuild at any time.

    Args:
        taxa_path: Path to the GBIF species-list CSV.
        taxa_cols: Column name list for the CSV.
        taxon_rank: Taxonomic rank string (e.g. "order", "species").
        device: Torch device string ("cpu" or "cuda").
        flag_the_det_errors: Unused; kept for API compatibility.

    Returns:
        TreeOfLifeClassifier with embeddings filtered to the taxa in taxa_path.
    """
    # No species list → use the full unfiltered Tree of Life.
    if not taxa_path or (isinstance(taxa_path, str) and not taxa_path.strip()):
        _ensure_hf_mode()
        print("No species list provided — loading full Tree of Life classifier (no taxon filter).")
        return TreeOfLifeClassifier(device=device)

    cache_path = os.path.splitext(taxa_path)[0] + ".pt"

    _ensure_hf_mode()

    # ── Fast path: load pre-filtered embeddings from cache ─────────────────
    if os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            cache = {}

        if "txt_embeddings" in cache and "txt_names" in cache:
            n = cache["txt_embeddings"].shape[1]
            print(f"Loading BioCLIP model with cached filter ({n} filtered labels) — skipping full TOL embedding load")
            classifier = _build_classifier_from_embedding_cache(
                cache["txt_embeddings"], cache["txt_names"], device
            )
            return classifier

        if "keep_labels_ary" in cache:
            print("ℹ️  Old-style bool-mask cache found — running full load once to upgrade cache format.")
        else:
            print("⚠️  Unrecognised cache — rebuilding from scratch.")

    # ── Slow path: load full TOL (~900 MB), filter, save v2 cache ──────────
    print("Loading TOL classifier (slow first run — result will be cached for next time)...")
    classifier = TreeOfLifeClassifier(device=device)

    taxon_keys = load_taxon_keys(
        taxa_path=taxa_path,
        taxa_cols=taxa_cols,
        taxon_rank=taxon_rank.lower(),
        flag_det_errors=flag_the_det_errors,
    )

    print(f"Filtering TOL embeddings to {len(taxon_keys)} {taxon_rank} values...")
    label_data = classifier.get_label_data()

    # Use isin() rather than create_taxa_filter() — GBIF lists contain taxa not
    # in TOL, and create_taxa_filter() raises on unknown values.
    keep_labels_ary = label_data[taxon_rank].str.lower().isin(taxon_keys).tolist()
    matched = sum(keep_labels_ary)
    print(f"Keeping {matched} of {len(keep_labels_ary)} TOL embeddings")

    if matched == 0:
        raise ValueError(
            f"No TOL embeddings matched the {taxon_rank} values in {taxa_path}. "
            "Check that the taxon_rank column name matches and the CSV contains valid taxa."
        )

    classifier.apply_filter(keep_labels_ary)

    # Save v2 cache: the filtered tensor + names so next run skips the full load.
    print(f"Saving filtered embedding cache to {cache_path}")
    torch.save(
        {
            "txt_embeddings": classifier._subset_txt_embeddings.cpu(),
            "txt_names": classifier._subset_txt_names,
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
            patchfullpath = (
                resolve_patch_path(thepatch_list[idx], image_path, DATASET_ROOT)
                if DATASET_ROOT
                else os.path.dirname(image_path) + "/" + thepatch_list[idx]
            )
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


def _id_checkpoint_dir(matched_img_json_pairs, label):
    """Return the per-run checkpoint directory for crash-resume of overwrite runs.

    Keyed by the dataset (its JSON paths) plus everything that determines the
    result — model VERSION, species-list DOI, taxonomic rank, and HU/BOT label.
    Changing any of those yields a different directory, so a genuinely different
    job is treated as fresh rather than resumed.

    The directory holds a single ``done.txt`` listing the representatives already
    written to disk. It survives a crash and is deleted on successful completion,
    which is precisely what lets an *overwrite* run tell "interrupted, resume"
    (directory present) from "fresh overwrite, redo everything" (absent).
    """
    key_parts = [label, VERSION, DOI, str(TAXONOMIC_RANK_FILTER)]
    key_parts += sorted(str(jp) for _, jp in matched_img_json_pairs)
    digest = hashlib.md5("\n".join(key_parts).encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".mothbot" / "id_cache" / digest


def _rep_key(rep):
    """Stable identity of a representative for the checkpoint: its JSON + shape idx."""
    return f"{rep[1]}\t{rep[2]}"


def _clear_id_checkpoint(ckpt_dir):
    """Delete a completed run's checkpoint so the next run starts fresh."""
    if ckpt_dir is None:
        return
    try:
        for f in ckpt_dir.iterdir():
            f.unlink()
        ckpt_dir.rmdir()
    except OSError:
        pass


def run_id_on_detection_set(matched_img_json_pairs, classifier, label):
    """Collect, cluster-deduplicate, then chunk-predict-and-bank IDs for one set.

    Rather than calling classifier.predict() once on every representative (which
    forces BioCLIP to do one large, invisible result-aggregation at the very end
    and writes nothing to disk until it finishes), we predict in chunks of
    PREDICT_CHUNK. After each chunk we immediately write its IDs to the JSON files
    on disk. This gives three things the single-call version could not:

      1. Live progress — each chunk reports throughput and a real ETA, and the
         predict() callback reports sub-chunk progress so the console never sits
         silent for minutes.
      2. Crash-safe banking — completed chunks are already on disk, so a crash at
         minute 100 of a 200-minute run loses at most one chunk's work.
      3. Resumability — see the checkpoint handling below. A crashed run resumes
         from where it stopped; a fresh overwrite run redoes everything.

    Resume vs. overwrite:
      • When NOT overwriting, collect_patches already dropped anything previously
        ID'd, so a crashed run resumes naturally (shapes we rewrote before the
        crash now count as pre-ID'd) — no checkpoint needed.
      • When overwriting, we intentionally re-ID existing shapes, so JSON state
        can't tell "done this run" from "done a previous run". Instead we record
        completed representatives in a checkpoint file that survives a crash and
        is deleted on success. A fresh overwrite run has no checkpoint and so
        re-IDs everything; only an interrupted one skips already-banked reps.

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

    # ── Resume handling (only meaningful when overwriting; see docstring) ──
    ckpt_dir = None
    done_keys = set()
    if OVERWRITE_EXISTING_IDs:
        ckpt_dir = _id_checkpoint_dir(matched_img_json_pairs, label)
        done_file = ckpt_dir / "done.txt"
        if done_file.exists():
            # An earlier run of this exact job crashed — resume from its checkpoint.
            done_keys = {ln for ln in done_file.read_text().splitlines() if ln}
        else:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    reps_to_do, members_to_do, already_done = [], [], 0
    for rep, members in zip(representatives, cluster_members):
        if done_keys and _rep_key(rep) in done_keys:
            already_done += 1
        else:
            reps_to_do.append(rep)
            members_to_do.append(members)

    if already_done:
        print(
            f"  🔄 Resuming an interrupted run — {already_done} of "
            f"{len(representatives)} representatives already banked; skipping them."
        )

    total_to_do = len(reps_to_do)
    if total_to_do == 0:
        print(f"  ✅ All {label} representatives already identified — nothing to do.")
        _clear_id_checkpoint(ckpt_dir)
        return

    print(
        f"  Running BioCLIP on {total_to_do} representative images "
        f"(from {len(all_patches)} detections) in chunks of {PREDICT_CHUNK}, "
        f"banking each chunk to disk as it completes..."
    )

    batch_size = 32 if torch.cuda.is_available() else 8
    rank_label = str(TAXONOMIC_RANK_FILTER.get_label())
    start_time = time.time()
    processed = 0          # representatives predicted so far
    written_dets = 0       # detections (cluster members) written so far
    pending_imgs, pending_members = [], []

    def flush_chunk():
        """Predict on the accumulated chunk, write every result to disk, report."""
        nonlocal processed, written_dets
        if not pending_imgs:
            return
        base = processed  # representatives finished before this chunk

        def chunk_callback(done_in_chunk, total_in_chunk):
            # Sub-chunk heartbeat so long chunks never sit silent.
            if done_in_chunk > 0 and (
                done_in_chunk % (batch_size * 5) == 0 or done_in_chunk == total_in_chunk
            ):
                elapsed = time.time() - start_time
                overall = base + done_in_chunk
                rate = overall / elapsed if elapsed > 0 else 0
                remaining = (total_to_do - overall) / rate if rate > 0 else 0
                print(
                    f"   🧠 {overall}/{total_to_do} IDs — "
                    f"{rate:.1f} IDs/s — ~{remaining/60:.1f} min remaining"
                )

        preds = list(classifier.predict(
            pending_imgs, rank=TAXONOMIC_RANK_FILTER, k=1,
            batch_size=batch_size, callback=chunk_callback,
        ))

        chunk_dets = 0
        done_this_chunk = []
        for members, pred in zip(pending_members, preds):
            winner = pred.get(rank_label, "")
            winnerprob = pred.get("score", 0.0)
            # Write the representative (members[0]) LAST so that a rep recorded in
            # the checkpoint reliably means the whole cluster was banked — even if
            # a crash interrupts this cluster's write loop.
            ordered = members[1:] + members[:1]
            apply_id_to_cluster(
                [m[1] for m in ordered], [m[2] for m in ordered],
                winner, winnerprob, pred,
            )
            chunk_dets += len(members)
            done_this_chunk.append(_rep_key(members[0]))

        # Record this chunk's reps AFTER their JSONs are written, so a crash can
        # only ever under-count what's done (harmless re-work), never over-count.
        # The checkpoint is best-effort: the real output is the JSON files, which
        # are already on disk, so a checkpoint-write failure must never crash the
        # ID run. Recreate the dir first in case it was removed mid-run (e.g. a
        # concurrent/leftover ID process cleared its own completed checkpoint).
        if ckpt_dir is not None:
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                with open(ckpt_dir / "done.txt", "a") as f:
                    f.write("".join(k + "\n" for k in done_this_chunk))
            except OSError as e:
                print(f"  ⚠️ Could not update resume checkpoint ({e}); "
                      f"continuing — IDs for this chunk are already saved.")

        for im in pending_imgs:
            try:
                im.close()
            except Exception:
                pass

        processed += len(pending_imgs)
        written_dets += chunk_dets
        pending_imgs.clear()
        pending_members.clear()
        print(
            f"   💾 Banked chunk to disk — {processed}/{total_to_do} representative "
            f"IDs done ({written_dets} detections written so far)."
        )

    for rep, members in zip(reps_to_do, members_to_do):
        try:
            pending_imgs.append(Image.open(rep[0]))
            pending_members.append(members)
        except Exception as e:
            print(f"  ⚠️ Could not open representative {rep[0]}: {e}")
            continue
        if len(pending_imgs) >= PREDICT_CHUNK:
            flush_chunk()
    flush_chunk()  # final partial chunk

    # Run finished cleanly — drop the checkpoint so the next run starts fresh
    # (this is what makes a later overwrite run re-ID everything).
    _clear_id_checkpoint(ckpt_dir)

    total_time = time.time() - start_time
    print(
        f"✅ {label} ID complete — {written_dets} detections identified via "
        f"{processed} representative IDs in {total_time:.1f}s ({total_time/60:.1f} min)"
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


def _ensure_hf_mode():
    """Disable HuggingFace Hub network access if no internet is available.

    TreeOfLifeClassifier() triggers huggingface_hub to HEAD-check its embeddings
    file on every init, even when they are already cached. With no connection this
    causes ~40 s of retries. A 2-second DNS probe avoids that wait.
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("8.8.8.8", 53))
        sock.close()
    except OSError:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("No internet connection detected — HuggingFace Hub will use cached files only.")


def extract_doi_from_csv_path(csv_path: str) -> str:
    """
    Extracts the DOI from a filename like:
      SpeciesList_..._doi.org10.15468dl.epzeza.csv       (old: no separator dot)
      SpeciesList_..._doi.org.10.15468.dl.pkbmuj.csv     (new: dot-separated)
    and returns the full DOI URL.
    Returns "no_doi" if no valid DOI is found.
    """
    filename = os.path.basename(csv_path)

    match = re.search(r"(doi\.org[0-9A-Za-z\.\-]+)", filename)
    if not match:
        return "no_doi"

    doi_raw = match.group(1)
    # Strip "doi.org", any leading separator dot, and trailing ".csv" if the
    # regex greedily consumed it as part of the DOI chunk.
    doi_core = doi_raw.replace("doi.org", "").lstrip(".")
    if doi_core.lower().endswith(".csv"):
        doi_core = doi_core[:-4]

    # DOI prefix is always "10.<registrant>" — find the split point.
    m = re.match(r"(10\.\d+)(.+)", doi_core)
    if not m:
        return "no_doi"

    prefix, suffix = m.groups()
    # suffix may start with "." in dot-separated filenames — strip it.
    doi_fixed = f"{prefix}/{suffix.lstrip('.')}"

    return f"https://doi.org/{doi_fixed}"


def run(
    input_path, taxa_csv, rank=3, ID_Hum=True, ID_Bot=True, overwrite_prev_bot_ID=True,
    dataset_root=None,
):
    """Run the full ID pipeline programmatically.

    Parameters
    ----------
    input_path : str
        Root folder containing mothbox data (any sub-folder structure).
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
    dataset_root : str | None
        Top-level folder for the _processed output tree.  Defaults to
        *input_path* itself.
    """
    global TAXONOMIC_RANK_FILTER, OVERWRITE_EXISTING_IDs, ID_HUMANDETECTIONS
    global ID_BOTDETECTIONS, INPUT_PATH, DOI, DEVICE, DATASET_ROOT

    TAXONOMIC_RANK_FILTER = Rank(int(rank))
    OVERWRITE_EXISTING_IDs = bool(overwrite_prev_bot_ID)
    ID_HUMANDETECTIONS = bool(ID_Hum)
    ID_BOTDETECTIONS = bool(ID_Bot)
    INPUT_PATH = input_path
    DATASET_ROOT = dataset_root or input_path

    DOI = extract_doi_from_csv_path(taxa_csv)
    print("using species list: " + DOI)

    DEVICE = get_device()

    # TODO: Re-enable once pybioclip CUDA performance is fixed.
    #print("Note: CUDA temporarily disabled for ID while we figure out what's going on with bioclip and CUDA")
    #DEVICE = "cpu"

    print_device_info(selected_device=DEVICE)

    print("Looking in this folder for MothboxData: " + INPUT_PATH)

    # Use structure-agnostic discovery: finds JSONs in the _processed tree
    hu_matched_img_json_pairs, bot_matched_img_json_pairs = (
        find_detection_matches_processed(DATASET_ROOT, source_folder=input_path)
    )

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