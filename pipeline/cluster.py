#!/usr/bin/env python3

"""
Mothbot_Cluster

This script tries to group all the detections in a night perceptually and then temporally

It takes a path to a nightly folder containing already detected creatures

Long Description:
The clustering algorithm aims to group detections by visual similarity. It achieves this grouping in the following way. First it extracts the embeddings of each of the night's visual detections using DINOv2. The Open Source DINOv2 is a machine learning model for "producing universal features suitable for image-level visual tasks" [https://dinov2.metademolab.com/] and is used to abstract each image into a set of visual features in a hyper-dimensional parameter space. (In the case that DINOv2 is not accessible for whatever reason, the script also defaults to just extracting features via a more basic histogram approach). 
Once all the embeddings of the features have been loaded into a common hyperspace, we group images with visual similarity using HDBSCAN. This open source python library which stands for Hierarchical Density-Based Spatial Clustering of Applications with Noise [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html], allows us to quickly find groupings within arbitrary data.  Images that are grouped into clusters of at least 2 other detections are assigned a unique positive "Visual Cluster" number such as "3" or "56". Detections with no other visual mates are given the same "-1" "Visual Cluster" identifier, indicating that they are somewhat unique within this dataset (and thus likely a unique insect). 
Then we further organize these visual clusters through a stage of "temporal clustering," where we pass over these visual clusters and see if they have neighbors in adjacent source images. If so, temporal lineages , or tracks within visual clusters are given a sub-designation with an extra numerical suffix (such as "3.1", "3.2", "3.17). 
This clustering then allows for rapid selection of visually similar detections in our Classify UI.



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
    find_detection_matches_processed,
    find_images_recursive,
    update_main_list,
    current_timestamp,
    get_rotated_rect_raw_coordinates,
    get_device,
    print_device_info,
)
from core.paths import resolve_patch_path, get_processed_folder


# ~~~~Variables to Change~~~~~~~

INPUT_PATH = r"C:\Users\andre\Desktop\donald\2022-01-11"  # raw string

# you probably always want these below as true
ID_HUMANDETECTIONS = True
ID_BOTDETECTIONS = True
DATASET_ROOT = None  # Set by run(); used for patch path resolution

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
    n = len(embeddings)

    # --- L2-normalize embeddings ---
    # DINOv2 features are most meaningful when compared by direction (cosine
    # similarity), not by magnitude.  L2-normalizing maps all vectors onto the
    # unit hypersphere so euclidean distance == sqrt(2*(1-cosine_sim)), giving
    # distances a consistent 0–2 scale regardless of batch size or model output.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    # --- PCA for large datasets ---
    # HDBSCAN's 'generic' fallback computes a full n×n pairwise distance matrix
    # when dimensionality is high (384 dims triggers this path).  For n=50,000
    # that is ~10 GB and kills the process.  PCA reduces to 50 dims where
    # HDBSCAN can use its boruvka_balltree path (~50 MB total) instead.
    # sklearn PCA uses randomized SVD, so memory peaks at ~200 MB during the
    # reduction regardless of dataset size.
    _LARGE_N = 10_000
    algorithm = "best"
    if n > _LARGE_N:
        from sklearn.decomposition import PCA
        n_pca = 50
        print(
            f"  Large dataset ({n:,} images) — reducing {embeddings.shape[1]}-dim embeddings "
            f"to {n_pca} dims via PCA to avoid HDBSCAN memory spike..."
        )
        pca = PCA(n_components=n_pca, random_state=42)
        embeddings = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_.sum()
        # Re-L2-normalize so epsilon distances remain in [0, 2]
        norms2 = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms2 = np.where(norms2 == 0, 1.0, norms2)
        embeddings = embeddings / norms2
        print(f"  PCA complete — {explained:.1%} of variance retained. Clustering in {n_pca}-dim space...")
        algorithm = "boruvka_balltree"

    # --- min_cluster_size ---
    # Fixed at 2: any pair of visually similar images forms a cluster.
    # Using 3 on large datasets was the main cause of ~30% of obvious matches
    # being left as noise (-1) — two identical-looking moths with no third
    # similar image would both be discarded as unclustered.
    min_cluster_size = 2

    # --- cluster_selection_epsilon ---
    # After L2-normalization, distances are bounded [0, 2].  epsilon=0.42
    # corresponds roughly to cosine_similarity ≥ 0.82, which handles real-world
    # variation in pose, scale, and trap lighting between moths of the same
    # species.  0.4 was still leaving large swaths of visually identical moths
    # unclustered; 0.6 was too broad
    # species in separate clusters.
    epsilon = 0.42

    # --- cluster_selection_method ---
    # "leaf" keeps the finest-grained clusters, giving more clusters and
    # better separation for diverse insect collections.
    cluster_selection_method = "leaf"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
        algorithm=algorithm,
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters = len(unique_labels)

    print(
        f"✅ The clusterer (HDBSCAN) created {n_clusters} clusters of similar insect photos "
        f"(and {np.sum(labels == -1)} noise points — i.e. visually unique insects)."
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
    pattern_C = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}[+-]\d{2}-\d{2})")  # ISO 8601 filename-safe

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
            elif match_B := pattern_B.search(fname):
                ts_str = match_B.group(1)
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")

            # Try Scheme C (new ISO 8601 filename-safe format)
            elif match_C := pattern_C.search(fname):
                ts_str = match_C.group(1)
                # Parse datetime portion only — strip the UTC offset for naive comparison
                # (all devices in a single night's data share the same offset, so naive
                # local time is still correct for temporal proximity sorting)
                ts = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H-%M-%S")

            if ts is None:
                print(f"⚠️  Could not parse timestamp from filename: {fname} — skipping temporal sub-clustering for this detection.")
                timestamps.append((i, None))
                continue

            timestamps.append((i, ts))

        # Sort detections in this cluster by time — drop any with unparseable timestamps
        timestamps = [(i, ts) for (i, ts) in timestamps if ts is not None]
        if not timestamps:
            continue
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
                    # For external collections the patch lives in the same
                    # folder as the JSON, not in a mirrored tree.
                    json_dir = os.path.dirname(json_path)
                    direct_patch = os.path.join(json_dir, os.path.basename(thepatch_list[idx]))
                    if os.path.isfile(direct_patch):
                        patchfullpath = direct_patch
                    elif DATASET_ROOT:
                        patchfullpath = resolve_patch_path(thepatch_list[idx], image_path, DATASET_ROOT)
                    else:
                        patchfullpath = os.path.dirname(image_path) + "/" + thepatch_list[idx]

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
                    # For external collections the patch lives flat in the same
                    # folder as the JSON.  Check there first before trying the
                    # _processed mirror tree.
                    json_dir = os.path.dirname(json_path)
                    direct_patch = os.path.join(json_dir, os.path.basename(thepatch_list[idx]))
                    if os.path.isfile(direct_patch):
                        patchfullpath = direct_patch
                    elif DATASET_ROOT:
                        patchfullpath = resolve_patch_path(thepatch_list[idx], image_path, DATASET_ROOT)
                    else:
                        patchfullpath = os.path.dirname(image_path) + "/" + thepatch_list[idx]

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


def _is_external_collection(input_path, dataset_root):
    """Return True if *input_path* should be treated as an externally-processed
    collection — i.e. it contains .jpg patch images but has no paired JSON
    detection files in the _processed mirror (or next to the images).

    This covers two cases:
    1. The folder is inside a ``_processed`` tree (collaborator formatted their
       data using our mirror layout).
    2. The folder contains jpgs but none of them have a corresponding
       ``_botdetection.json`` or ``.json`` in the expected output location.
    """
    from core.paths import get_json_output_path

    # Case 1: folder is already inside a _processed tree
    parts = os.path.normpath(input_path).split(os.sep)
    if "_processed" in parts:
        return True

    # Case 2: scan all jpgs and check whether ANY have a processed JSON
    jpgs = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(".jpg")
    ]
    if not jpgs:
        return False  # no jpgs at all — not an external patch folder

    for jpg in jpgs:
        # Check _processed mirror location
        try:
            bot_json = get_json_output_path(jpg, "_botdetection", dataset_root)
            hu_json  = get_json_output_path(jpg, "", dataset_root)
        except ValueError:
            continue
        if os.path.isfile(bot_json) or os.path.isfile(hu_json):
            return False  # at least one processed JSON exists — normal collection
        # Also check next to the source image (legacy human ground-truth)
        if os.path.isfile(jpg.replace(".jpg", "_botdetection.json")) or \
           os.path.isfile(jpg.replace(".jpg", ".json")):
            return False

    # No JPGs have any associated JSON anywhere — treat as external
    return True


def _build_stub_jsons(folder):
    """Reverse-build minimal bot-detection JSON stubs from patch images in
    *folder* so that Cluster can operate on externally-processed collections
    that arrived without JSON files.

    Patch filename format assumed:  <source_stem>_<detidx>_<modelname>.jpg
    One stub JSON is written per inferred source image stem.

    Returns the number of stubs created.
    """
    import json as _json
    from pathlib import Path as _Path

    folder = _Path(folder)
    patch_files = sorted(folder.glob("*.jpg"))
    groups = {}
    for pf in patch_files:
        parts = pf.stem.rsplit("_", 2)
        source_stem = "_".join(parts[:-2]) if len(parts) >= 3 else pf.stem
        groups.setdefault(source_stem, []).append(pf)

    created = 0
    for source_stem, patches in groups.items():
        json_path = folder / f"{source_stem}_botdetection.json"
        if json_path.exists():
            continue
        shapes = [
            {
                "label": "creature",
                "points": [],
                "patch_path": pf.name,
                "confidence_detection": None,
                "identifier_bot": "",
                "identifier_human": "",
                "timestamp_detection": "",
                "detector_bot": "external",
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "score": None,
                "direction": 0,
                "group_id": None,
                "description": "",
                "difficult": "false",
                "kie_linking": [],
            }
            for pf in sorted(patches)
        ]
        stub = {
            "version": "external",
            "flags": {},
            "imagePath": source_stem + ".jpg",
            "imageHeight": None,
            "imageWidth": None,
            "description": "stub generated from external patches",
            "imageData": None,
            "shapes": shapes,
        }
        with open(json_path, "w") as fh:
            _json.dump(stub, fh, indent=4)
        created += 1

    return created


def _find_pairs_in_external_folder(folder):
    """Find (patch_placeholder, json_path) pairs directly inside *folder*.

    For external collections there are no original source images — the patches
    ARE the images.  We use the stub JSONs (or any *_botdetection.json) and
    pair each with a synthetic image path equal to the JSON's imagePath field
    (even if that file doesn't exist on disk — Cluster only needs the patch
    paths stored inside the JSON, not the source image itself).

    Returns (hu_pairs, bot_pairs) in the same format as
    find_detection_matches_processed.
    """
    import json as _json
    from pathlib import Path as _Path

    folder = _Path(folder)
    bot_pairs = []
    hu_pairs = []

    for jf in sorted(folder.glob("*_botdetection.json")):
        try:
            data = _json.loads(jf.read_text())
            image_path = str(folder / data.get("imagePath", jf.stem.replace("_botdetection", "") + ".jpg"))
            bot_pairs.append((image_path, str(jf)))
        except Exception:
            continue

    for jf in sorted(folder.glob("*.json")):
        if jf.name.endswith("_botdetection.json"):
            continue
        try:
            data = _json.loads(jf.read_text())
            image_path = str(folder / data.get("imagePath", jf.stem + ".jpg"))
            hu_pairs.append((image_path, str(jf)))
        except Exception:
            continue

    return hu_pairs, bot_pairs


def run(input_path, ID_Hum=True, ID_Bot=True, dataset_root=None):
    """Entry point for clustering detections (callable from other modules).

    Parameters
    ----------
    input_path : str
        Root folder containing detection data (any sub-folder structure).
        May be a flat folder of patch images from an external collaborator —
        in that case stub JSON files are auto-created from the patch filenames
        before clustering proceeds.
    ID_Hum : bool
        Process human-annotated detections.
    ID_Bot : bool
        Process bot detections.
    dataset_root : str | None
        Top-level folder for the _processed output tree.  Defaults to
        *input_path* itself.
    """
    global INPUT_PATH, ID_HUMANDETECTIONS, ID_BOTDETECTIONS, DEVICE, DATASET_ROOT

    INPUT_PATH = input_path
    ID_HUMANDETECTIONS = ID_Hum
    ID_BOTDETECTIONS = ID_Bot
    DATASET_ROOT = dataset_root or input_path

    print("Starting script to cluster detections into meaningful groups")

    DEVICE = get_device()
    print_device_info(selected_device=DEVICE)

    # ~~~~~~~~~~~~~~~~ GATHERING DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Looking in this folder for MothboxData: " + INPUT_PATH)

    # Detect whether this is an externally-processed collection.
    # Definition: input_path contains .jpg files AND none of those jpgs have a
    # paired _botdetection.json or .json sitting in the _processed mirror (or
    # next to them).  We check the whole folder, not just a spot-sample.
    is_external = _is_external_collection(input_path, DATASET_ROOT)

    if is_external:
        print(f"External patch collection detected in: {input_path}")
        created = _build_stub_jsons(input_path)
        print(f"Built {created} stub detection JSON(s) from patch filenames.")
        hu_matched_img_json_pairs, bot_matched_img_json_pairs = _find_pairs_in_external_folder(input_path)
        # For external collections the patches live flat in input_path itself.
        # Set DATASET_ROOT = input_path so that resolve_patch_path resolves
        # patch filenames relative to that folder (not a nested _processed tree).
        DATASET_ROOT = input_path
    else:
        hu_matched_img_json_pairs, bot_matched_img_json_pairs = (
            find_detection_matches_processed(DATASET_ROOT, source_folder=input_path)
        )

    print(
        "Found ",
        str(len(hu_matched_img_json_pairs))
        + " pairs of images and HUMAN detection data to try to cluster",
    )
    if len(hu_matched_img_json_pairs) > 0:
        print("example human detection and json pair:")
        print(hu_matched_img_json_pairs[0])

    print(
        "Found ",
        str(len(bot_matched_img_json_pairs))
        + " pairs of images and BOT detection data to try to cluster",
    )
    if len(bot_matched_img_json_pairs) > 0:
        print("example bot detection and json pair:")
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