#!/usr/bin/env python3
"""
Mothbot processed-output path helpers.

All pipeline outputs (JSON detection files, patch thumbnails, CSV exports,
FiftyOne datasets) are written into a mirrored ``_processed`` tree that sits
alongside the raw data folder chosen by the user.  The raw source images are
never touched or moved.

Layout
------
Given a user-chosen *dataset root* such as::

    /data/MyDataset/

The processed mirror is::

    /data/MyDataset/_processed/

Every sub-folder that exists under the dataset root is mirrored there.
Patches live **flat** in the same mirrored folder as the JSON files —
no ``patches/`` sub-folder::

    /data/MyDataset/Deploy_A/2025-06-21/CAM_img.jpg          ← raw image, untouched
    /data/MyDataset/_processed/Deploy_A/2025-06-21/CAM_img_botdetection.json
    /data/MyDataset/_processed/Deploy_A/2025-06-21/CAM_img_0_Mothbot_model.jpg

Public API
----------
get_processed_root(dataset_root)
    Return the ``_processed`` folder path for *dataset_root*.

get_processed_folder(source_folder, dataset_root)
    Mirror *source_folder* (which must be inside *dataset_root*) into the
    ``_processed`` tree and return the mirrored path.

get_json_output_path(source_image_path, suffix, dataset_root)
    Return the path where a JSON file produced from *source_image_path*
    should be written.  *suffix* is appended before ``.json``, e.g.
    ``"_botdetection"`` → ``DEVICE_..._botdetection.json``.

get_patch_output_path(source_image_path, det_idx, model_name, dataset_root)
    Return the path where a patch image for a specific detection should
    be written.

resolve_patch_path(patch_rel, source_image_path, dataset_root)
    Given the patch filename stored inside a JSON detection file and the
    original source image path, return the absolute path to that patch
    in the ``_processed`` tree.
"""

import os
from pathlib import Path

_PROCESSED_DIR_NAME = "_processed"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_processed_root(dataset_root: str) -> str:
    """Return the ``_processed`` folder path for *dataset_root*.

    The folder is created on first access.
    """
    processed = os.path.join(dataset_root, _PROCESSED_DIR_NAME)
    Path(processed).mkdir(parents=True, exist_ok=True)
    return processed


def get_processed_folder(source_folder: str, dataset_root: str) -> str:
    """Mirror *source_folder* into the ``_processed`` tree.

    *source_folder* must be at or below *dataset_root*.  Returns the
    corresponding path inside ``_processed/`` and creates it if needed.

    Example
    -------
    >>> get_processed_folder(
    ...     "/data/D/Deploy_A/2025-06-21",
    ...     "/data/D",
    ... )
    '/data/D/_processed/Deploy_A/2025-06-21'
    """
    dataset_root = os.path.realpath(dataset_root)
    source_folder = os.path.realpath(source_folder)

    try:
        rel = os.path.relpath(source_folder, dataset_root)
    except ValueError:
        # On Windows, relpath raises ValueError when paths are on different drives.
        raise ValueError(
            f"source_folder '{source_folder}' is not inside dataset_root '{dataset_root}'"
        )

    # Guard against accidentally escaping the dataset root with ".."
    if rel.startswith(".."):
        raise ValueError(
            f"source_folder '{source_folder}' is not inside dataset_root '{dataset_root}'"
        )

    # Strip a leading "_processed" component to prevent double-nesting when
    # the caller accidentally passes a path that is already in the mirror tree.
    parts = Path(rel).parts
    if parts and parts[0] == _PROCESSED_DIR_NAME:
        rel = str(Path(*parts[1:])) if len(parts) > 1 else "."

    processed_folder = os.path.join(dataset_root, _PROCESSED_DIR_NAME, rel)
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    return processed_folder


def get_json_output_path(
    source_image_path: str, suffix: str, dataset_root: str
) -> str:
    """Return the output path for a JSON file derived from *source_image_path*.

    Parameters
    ----------
    source_image_path:
        Absolute path to the raw source ``.jpg`` image.
    suffix:
        String appended before ``.json``, e.g. ``""`` for human ground-truth
        or ``"_botdetection"`` for YOLO output.
    dataset_root:
        The top-level folder the user chose to process.

    Example
    -------
    >>> get_json_output_path(
    ...     "/data/D/Deploy_A/2025-06-21/CAM_2025-06-21-00-00-00_HDR0.jpg",
    ...     "_botdetection",
    ...     "/data/D",
    ... )
    '/data/D/_processed/Deploy_A/2025-06-21/CAM_2025-06-21-00-00-00_HDR0_botdetection.json'
    """
    source_folder = os.path.dirname(source_image_path)
    processed_folder = get_processed_folder(source_folder, dataset_root)
    basename = os.path.basename(source_image_path)
    stem = basename[: basename.rfind(".")] if "." in basename else basename
    return os.path.join(processed_folder, stem + suffix + ".json")


def get_patch_output_path(
    source_image_path: str, det_idx: int, model_name: str, dataset_root: str
) -> str:
    """Return the output path for a detection patch image.

    Patches live flat in the same mirrored folder as the JSON files — no
    ``patches/`` sub-folder.

    Example
    -------
    >>> get_patch_output_path(
    ...     "/data/D/Deploy_A/2025-06-21/CAM_img.jpg",
    ...     0,
    ...     "Mothbot_model.pt",
    ...     "/data/D",
    ... )
    '/data/D/_processed/Deploy_A/2025-06-21/CAM_img_0_Mothbot_model.pt.jpg'
    """
    source_folder = os.path.dirname(source_image_path)
    processed_folder = get_processed_folder(source_folder, dataset_root)
    basename = os.path.basename(source_image_path)
    stem, ext = (basename.rsplit(".", 1) if "." in basename else (basename, "jpg"))
    patch_filename = f"{stem}_{det_idx}_{model_name}.{ext}"
    return os.path.join(processed_folder, patch_filename)


def resolve_patch_path(
    patch_rel: str, source_image_path: str, dataset_root: str
) -> str:
    """Resolve a patch filename stored in a JSON detection file to its
    absolute location in the ``_processed`` mirror tree.

    Parameters
    ----------
    patch_rel:
        The ``patch_path`` value from a JSON shape.  May be a bare filename
        (``"CAM_img_0_Mothbot.jpg"``) or the legacy ``"patches/<filename>"``
        format — both are handled.
    source_image_path:
        Absolute path to the raw source image that produced the detection.
    dataset_root:
        The top-level folder the user chose to process.
    """
    source_folder = os.path.dirname(source_image_path)
    processed_folder = get_processed_folder(source_folder, dataset_root)
    # Strip any leading "patches/" prefix from the legacy format
    patch_filename = os.path.basename(patch_rel)
    return os.path.join(processed_folder, patch_filename)


# ---------------------------------------------------------------------------
# Convenience: scan for existing JSON outputs in the _processed mirror
# ---------------------------------------------------------------------------

def find_processed_json(
    source_image_path: str, suffix: str, dataset_root: str
) -> str | None:
    """Return the JSON output path if it already exists, else None."""
    path = get_json_output_path(source_image_path, suffix, dataset_root)
    return path if os.path.isfile(path) else None
