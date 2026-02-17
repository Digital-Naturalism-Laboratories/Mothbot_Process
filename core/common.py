#!/usr/bin/env python3
"""
Shared utilities for Mothbot pipeline scripts.

Centralises functions that were duplicated across multiple worker scripts
(find_date_folders, find_detection_matches, scan_for_images, etc.) and
provides a lightweight stdout-capture helper used by the Gradio UI to
stream output from in-process worker calls.
"""

import io
import json
import os
import queue
import re
import sys
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# Folder / file discovery
# ---------------------------------------------------------------------------

# Matches "YYYY-MM-DD" or "Prefix_YYYY-MM-DD"
NIGHTLY_REGEX = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}|[A-Za-z0-9]+_\d{4}-\d{2}-\d{2})$"
)


def find_date_folders(directory):
    """Recursively find date-formatted folders (YYYY-MM-DD or Prefix_YYYY-MM-DD)."""
    folders = []
    if NIGHTLY_REGEX.match(os.path.basename(directory)):
        folders.append(directory)
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            if NIGHTLY_REGEX.match(d):
                folders.append(os.path.join(root, d))
    return sorted(folders)


def scan_for_images(folder_path):
    """Return sorted list of .jpg file paths in *folder_path*."""
    return sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".jpg")
    )


def find_detection_matches(folder_path):
    """Find matching (jpg, json) pairs for human and bot detections.

    Returns
    -------
    hu_matches : list[tuple[str, str]]
        (jpg_path, human_json_path) pairs.
    bot_matches : list[tuple[str, str]]
        (jpg_path, bot_json_path) pairs.
    """
    jpg_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".jpg")
    ]
    json_set = set(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".json")
    )

    hu_matches = []
    bot_matches = []
    for jpg in jpg_files:
        human_json = jpg.replace(".jpg", ".json")
        bot_json = jpg.replace(".jpg", "_botdetection.json")
        if human_json in json_set:
            hu_matches.append((jpg, human_json))
        if bot_json in json_set:
            bot_matches.append((jpg, bot_json))
    return hu_matches, bot_matches


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def update_main_list(main_list, new_items):
    """Append *new_items* to *main_list*, skipping duplicates."""
    existing = set(main_list)
    for item in new_items:
        if item not in existing:
            main_list.append(item)
            existing.add(item)
    return main_list


# ---------------------------------------------------------------------------
# Timestamp / device helpers
# ---------------------------------------------------------------------------

def current_timestamp():
    """Return current local timestamp as ``YYYY-MM-DD__HH_MM_SS_(+HHMM)``."""
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d__%H_%M_%S_(%z)")


def get_device():
    """Return ``'cuda'`` if a CUDA GPU is available, else ``'cpu'``."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_device_info():
    """Print CUDA availability and GPU details (or CPU fallback)."""
    import torch
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA not available, using CPU")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def get_rotated_rect_raw_coordinates(json_file):
    """Read rotated-rect coordinates, ID status, and patch paths from a
    detection JSON file.

    Returns
    -------
    coordinates_list : list
    pre_ided_list : list[bool]
    patch_list : list[str]
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    coordinates_list = []
    pre_ided_list = []
    patch_list = []
    pre_ided = False

    for shape in data["shapes"]:
        if shape["shape_type"] == "rotation":
            patch_list.append(shape["patch_path"])
            coordinates_list.append(shape["points"])
            if "identifier_bot" in shape and shape["identifier_bot"] != "":
                pre_ided = True
            pre_ided_list.append(pre_ided)

    return coordinates_list, pre_ided_list, patch_list


# ---------------------------------------------------------------------------
# Stdout capture for the Gradio UI (replaces subprocess streaming)
# ---------------------------------------------------------------------------

class _OutputCapture(io.TextIOBase):
    """File-like object that sends every ``write()`` to a ``queue.Queue``."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s) if s else 0

    def flush(self):
        pass


def run_in_thread(fn, *args, **kwargs):
    """Run *fn* in a background thread and **yield** captured stdout chunks.

    This is the primary mechanism for streaming worker output into a Gradio
    ``Textbox``.  Usage inside a Gradio event handler (which must be a
    generator)::

        output = ""
        for chunk in run_in_thread(Mothbot_Detect.run, input_path=folder, ...):
            output += chunk
            yield output          # Gradio updates the Textbox
    """
    cap = _OutputCapture()
    error_holder: list = [None]

    def _worker():
        _orig = sys.stdout
        sys.stdout = cap
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            sys.stdout = _orig
            cap.q.put(None)  # sentinel

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        try:
            chunk = cap.q.get(timeout=0.2)
        except queue.Empty:
            continue
        if chunk is None:
            break
        yield chunk

    t.join(timeout=10)
    if error_holder[0]:
        raise error_holder[0]
