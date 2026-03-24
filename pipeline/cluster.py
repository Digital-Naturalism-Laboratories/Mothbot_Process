#!/usr/bin/env python3

"""
Mothbot_Cluster

This script tries to group all the detections in a night perceptually and then temporally

It takes a path to a nightly folder containing already detected creatures


Usage:
  python Mothbox_ID.py

Arguments:
  -h, --help    Show this help message and exit

"""
import ssl
import timm

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # needed for some macs to automatically download files associated with some of the libraries
# import polars as pl
import os
import sys
import json
import argparse
import re
import inspect
import numpy as np
from PIL import Image
from PIL import ImageFile

# perception clustering
import torch
import torchvision.transforms as T
import sklearn.utils as sk_utils
from sklearn.utils import validation as sk_validation
from datetime import datetime, timedelta
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = (
    True  # makes ok for use images that are messed up slightly
)

# import PIL.Image
import warnings

warnings.filterwarnings("ignore", message="xFormers is not available*")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

# Compatibility shim for older third-party libraries (e.g. hdbscan) that still
# pass `force_all_finite` to scikit-learn's validation.check_array().
_check_array_sig = inspect.signature(sk_validation.check_array)
if (
    "force_all_finite" not in _check_array_sig.parameters
    and "ensure_all_finite" in _check_array_sig.parameters
):
    _original_check_array = sk_validation.check_array

    def _check_array_compat(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return _original_check_array(*args, **kwargs)

    sk_validation.check_array = _check_array_compat
    if hasattr(sk_utils, "check_array"):
        sk_utils.check_array = _check_array_compat

import hdbscan

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

INPUT_PATH = r"C:\Users\andre\Desktop\donald\2022-01-11"  # raw string

# you probably always want these below as true
ID_HUMANDETECTIONS = True
ID_BOTDETECTIONS = True

# Paths to save filtered list of embeddings/labels
image_embeddings_path = INPUT_PATH + "/image_embeddings.npy"
embedding_labels_path = INPUT_PATH + "/embedding_labels.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=False,
        default=INPUT_PATH,
        help="path to images for classification (ex: datasets/test_images/data)",
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

    return parser.parse_args()


# FUNCTIONS ~~~~~~~~~~~~~


####################################
# --------------------------
# # Perceptual Processing Functions
# --------------------------
####################################

# --------------------------
# 1. Lazy-load DINOv2 model
# --------------------------
_dino_model = None
_dino_transform = None


def _get_bundled_weights_path():
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        # cluster.py is in pipeline/, assets/ is one level up
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "dinov2_vits14_pretrain.pth")


def _ensure_dino_loaded():
    global _dino_model, _dino_transform
    if _dino_model is not None:
        return

    device = get_device()
    weights_path = _get_bundled_weights_path()

    if not os.path.exists(weights_path):
        raise RuntimeError(
            f"Bundled DINOv2 weights not found at: {weights_path}\n"
            "Please ensure dinov2_vits14_pretrain.pth is in the assets/ folder."
        )

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
    #model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, img_size=224) # this model gets grumpy if not 518
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    _dino_model = model

    _dino_transform = T.Compose([
        T.Resize(518),
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

# --------------------------
# 2. Extract embeddings
# --------------------------
def get_embedding(img_path):
    _ensure_dino_loaded()
    img = Image.open(img_path).convert("RGB")
    img_tensor = _dino_transform(img).unsqueeze(0).to(next(_dino_model.parameters()).device)
    with torch.no_grad():
        feat = _dino_model(img_tensor)
    return feat.cpu().numpy().squeeze()

def get_fallback_embedding(img_path):
    """Local deterministic embedding when DINOv2 hub cannot be used."""
    img = Image.open(img_path).convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Compact histogram embedding per-channel (32 bins x RGB = 96 dims).
    hist = []
    for channel in range(3):
        channel_hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0.0, 1.0))
        hist.append(channel_hist.astype(np.float32))
    feat = np.concatenate(hist)
    norm = np.linalg.norm(feat)
    return feat if norm == 0 else feat / norm

def extract_embeddings(image_files, batch_size=8):
    embeddings = []
    use_fallback = False
    try:
        _ensure_dino_loaded()
    except Exception as e:
        use_fallback = True
        print("⚠️ DINOv2 embedding model unavailable, falling back to histogram embeddings.")
        print(f"   details: {e}")

    device = None
    if not use_fallback:
        device = next(_dino_model.parameters()).device
    total = len(image_files)
    total_batches = (total + batch_size - 1) // batch_size
    print(f"🔍 Extracting embeddings for {total} images in {total_batches} batches on {device}...")

    import time
    start_time = time.time()

    for batch_num, i in enumerate(range(0, total, batch_size)):
        batch_paths = image_files[i:i+batch_size]

        if use_fallback:
            for path in batch_paths:
                try:
                    embeddings.append(get_fallback_embedding(path))
                except Exception as e:
                    print(f"⚠️ Skipping {path}: {e}")
        else:
            tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(_dino_transform(img))
                except Exception as e:
                    print(f"⚠️ Skipping {path}: {e}")
            if tensors:
                batch_tensor = torch.stack(tensors).to(device)
                with torch.no_grad():
                    feats = _dino_model(batch_tensor)
                embeddings.extend(feats.cpu().numpy())

        # Progress + ETA after first batch
        elapsed = time.time() - start_time
        batches_done = batch_num + 1
        images_done = min(i + batch_size, total)
        if batches_done == 1 and total_batches > 1:
            eta_seconds = (elapsed / batches_done) * (total_batches - batches_done)
            print(f"   ⏱️ First batch done in {elapsed:.1f}s — estimated {eta_seconds:.0f}s remaining ({eta_seconds/60:.1f} min)")
        elif batches_done % 5 == 0 or batches_done == total_batches:
            eta_seconds = (elapsed / batches_done) * (total_batches - batches_done)
            print(f"   📦 Batch {batches_done}/{total_batches} — {images_done}/{total} images — ~{eta_seconds:.0f}s remaining")

    total_time = time.time() - start_time
    print(f"✅ Embeddings complete — {total} images in {total_time:.1f}s ({total_time/60:.1f} min)")
    return np.array(embeddings)


# --------------------------
# 3. Cluster with HDBSCAN
# --------------------------
def cluster_embeddings(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,  # smaller clusters allowed
        min_samples=1,  # fewer items marked as noise
        cluster_selection_epsilon=0.05,  # expand clusters slightly
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    # Count clusters (ignore -1 which means "noise")
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters = len(unique_labels)

    print(
        f"✅ The clusterer (HDBSCAN() created {n_clusters} clusters of similar insect photos (and {np.sum(labels == -1)} noise points - ie insect photos that were unique)."
    )

    return labels


# --------------------------
# 4. Write cluster to JSON
# --------------------------
def write_cluster_to_json(filepaths, json_paths, idxes, labels):
    for fname, json_path, i, label in zip(filepaths, json_paths, idxes, labels):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if 0 <= i < len(data["shapes"]):
                shape = data["shapes"][i]
                shape["clusterID"] = float(label)
                shape["timestamp_cluster"] = current_timestamp()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            print(f"⚠️ Could not update {fname}: {e}")
    print("✅ Cluster IDs written into 'Json' field.")


# Subcluster through TIME
def temporal_subclusters(
    patch_paths_hu, json_paths_hu, idx_paths_hu, labels, gap_minutes=1
):
    """
    Creates temporal subclusters within perceptual clusters based on timestamp proximity.

    Args:
        patch_paths_hu (list[str]): Paths to parent images
        json_paths_hu (list[str]): Paths to JSON metadata
        idx_paths_hu (list[str]): Paths to cropped insect images
        labels (list[int]): Cluster IDs for each detection (from HDBSCAN etc.)
        gap_minutes (int, optional): Maximum gap (in minutes) allowed between
                                     consecutive detections in the same temporal chain.
                                     Default = 1.

    Returns:
        list[str]: A list of new cluster IDs (like "3.1", "3.2") aligned with inputs.
    """
    # Initialize result list (default keep -1 for noise)
    new_labels = [str(l) if l != -1 else "-1" for l in labels]

    # Group indices by cluster
    cluster_to_indices = defaultdict(list)
    for idx, cl in enumerate(labels):
        if cl != -1:  # skip noise
            cluster_to_indices[cl].append(idx)

    # Regex patterns for both schemes
    pattern_A = re.compile(
        r"(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})"
    )  # YYYY_MM_DD__HH_MM_SS
    pattern_B = re.compile(r"(\d{14})")  # YYYYMMDDHHMMSS

    for cluster_id, indices in cluster_to_indices.items():
        timestamps = []

        for i in indices:
            fname = os.path.basename(patch_paths_hu[i])

            ts_str = None
            ts = None

            # Try Scheme A
            match_A = pattern_A.search(fname)
            if match_A:
                ts_str = match_A.group(1)
                ts = datetime.strptime(ts_str, "%Y_%m_%d__%H_%M_%S")

            # Try Scheme B
            else:
                match_B = pattern_B.search(fname)
                if match_B:
                    ts_str = match_B.group(1)
                    ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")

            if ts is None:
                raise ValueError(f"Could not parse timestamp from filename: {fname}")

            timestamps.append((i, ts))

        # Sort detections in this cluster by time
        timestamps.sort(key=lambda x: x[1])

        # Find temporal sequences
        gap = timedelta(minutes=gap_minutes)
        seq_id = 1
        prev_time = None

        for i, ts in timestamps:
            if prev_time is None:
                # start first sequence
                new_labels[i] = f"{cluster_id}.{seq_id}"
                prev_time = ts
            else:
                if ts - prev_time <= gap:
                    # same sequence
                    new_labels[i] = f"{cluster_id}.{seq_id}"
                else:
                    # new sequence
                    seq_id += 1
                    new_labels[i] = f"{cluster_id}.{seq_id}"
                prev_time = ts

    return new_labels


# Maybe this?
def Cluster_matched_img_json_pairs(
    hu_matched_img_json_pairs, bot_matched_img_json_pairs, device
):

    # Process Human Detections
    print("processing Human Detections.........")
    patch_paths_hu = []  # define this once before your loop
    json_paths_hu = []
    idx_paths_hu = []

    if ID_HUMANDETECTIONS:
        # Next process each pair and generate temporary files for the ROI of each detection in each image
        # Iterate through image-JSON pairs
        index = 0
        numofpairs = len(hu_matched_img_json_pairs)
        for pair in hu_matched_img_json_pairs:

            # Load JSON file and extract rotated rectangle coordinates for each detection
            image_path, json_path = pair[:2]  # Always extract the first two elements

            coordinates_of_detections_list, was_pre_ided_list, thepatch_list = (
                get_rotated_rect_raw_coordinates(json_path)
            )
            index = index + 1
            print(
                str(index)
                + "/"
                + str(numofpairs)
                + "  | "
                + str(len(coordinates_of_detections_list)),
                "HUMAN detections in " + json_path,
            )
            if coordinates_of_detections_list:
                for idx, coordinates in enumerate(coordinates_of_detections_list):
                    # add path to list of patches for perceptual processing
                    patchfullpath = (
                        os.path.dirname(image_path) + "/" + thepatch_list[idx]
                    )

                    patch_paths_hu.append(patchfullpath)
                    json_paths_hu.append(json_path)
                    idx_paths_hu.append(idx)

    # Process BOT Detections
    print("processing BOT Detections.........")
    patch_paths_bots = []  # define this once before your loop
    json_paths_bots = []
    idx_paths_bots = []
    if ID_BOTDETECTIONS:
        # Next process each pair and generate temporary files for the ROI of each detection in each image
        # Iterate through image-JSON pairs
        index = 0
        numofpairs = len(bot_matched_img_json_pairs)
        for pair in bot_matched_img_json_pairs:

            # Load JSON file and extract rotated rectangle coordinates for each detection
            image_path, json_path = pair[:2]  # Always extract the first two elements

            coordinates_of_detections_list, was_pre_ided_list, thepatch_list = (
                get_rotated_rect_raw_coordinates(json_path)
            )
            index = index + 1
            print(
                str(index)
                + "/"
                + str(numofpairs)
                + "  | "
                + str(len(coordinates_of_detections_list)),
                "BOT detections in " + json_path,
            )
            if coordinates_of_detections_list:
                for idx, coordinates in enumerate(coordinates_of_detections_list):
                    patchfullpath = (
                        os.path.dirname(image_path) + "/" + thepatch_list[idx]
                    )

                    # add path to list of patches for later perceptual processing
                    patch_paths_bots.append(patchfullpath)
                    json_paths_bots.append(json_path)
                    idx_paths_bots.append(idx)

    # ~~~~~~~~~~~~~ PERCEPTUAL PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~
    # process perceptual similarities for bot and hu detections
    print("Loading Embeddings for Perceptual Processing...")
    batch_size = 32 if torch.cuda.is_available() else 8

    # Hu detections first
    if len(patch_paths_hu) > 0:
        embeddings = extract_embeddings(patch_paths_hu, batch_size=batch_size)
        labels = cluster_embeddings(embeddings)
        # save_clusters(input_folder, filenames, labels, output_folder)
        labels = temporal_subclusters(
            patch_paths_hu, json_paths_hu, idx_paths_hu, labels
        )
        write_cluster_to_json(patch_paths_hu, json_paths_hu, idx_paths_hu, labels)

    # bot detections first
    if len(patch_paths_bots) > 0:
        embeddings = extract_embeddings(patch_paths_bots,  batch_size=batch_size)
        labels = cluster_embeddings(embeddings)
        labels = temporal_subclusters(
            patch_paths_bots, json_paths_bots, idx_paths_bots, labels
        )
        write_cluster_to_json(patch_paths_bots, json_paths_bots, idx_paths_bots, labels)


def run(input_path, ID_Hum=True, ID_Bot=True):
    """Entry point for clustering detections (callable from other modules).

    Parameters
    ----------
    input_path : str
        Root folder containing dated sub-folders with detection data.
    ID_Hum : bool
        Process human-annotated detections.
    ID_Bot : bool
        Process bot detections.
    """
    global INPUT_PATH, ID_HUMANDETECTIONS, ID_BOTDETECTIONS, DEVICE

    INPUT_PATH = input_path
    ID_HUMANDETECTIONS = ID_Hum
    ID_BOTDETECTIONS = ID_Bot

    print("Starting script to cluster detections into meaningful groups")

    DEVICE = get_device()
    print_device_info(selected_device=DEVICE)

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
        + " pairs of images and HUMAN detection data to try to ID",
    )
    # Example Pair
    print("example human detection and json pair:")
    if len(hu_matched_img_json_pairs) > 0:
        print(hu_matched_img_json_pairs[0])

    print(
        "Found ",
        str(len(bot_matched_img_json_pairs))
        + " pairs of images and BOT detection data to try to ID",
    )
    # Example Pair
    print("example human detection and json pair:")
    if len(bot_matched_img_json_pairs) > 0:
        print(bot_matched_img_json_pairs[0])

    # ~~~~~~~~~~~~~~~~ Processing Data ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Cluster_matched_img_json_pairs(
        hu_matched_img_json_pairs,
        bot_matched_img_json_pairs,
        device=DEVICE,
    )

    print("Finished Automatic Clustering")


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input_path,
        ID_Hum=bool(int(args.ID_Hum)),
        ID_Bot=bool(int(args.ID_Bot)),
    )
