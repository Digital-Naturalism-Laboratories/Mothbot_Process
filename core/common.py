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
import logging
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


def find_images_recursive(dataset_root, processed_dir_name="_processed"):
    """Recursively find all .jpg files under *dataset_root*, skipping the
    ``_processed`` mirror tree and any ``patches/`` sub-folders.

    Returns a sorted list of absolute image paths.  This is the
    structure-agnostic replacement for ``find_date_folders`` + ``scan_for_images``
    and works regardless of how the user has organised their data.
    """
    images = []
    for root, dirs, files in os.walk(dataset_root):
        # Prune the _processed tree and patches folders so we never pick up
        # output artefacts as source images.
        dirs[:] = sorted(
            d for d in dirs
            if d != processed_dir_name and d.lower() != "patches"
        )
        for f in sorted(files):
            if f.lower().endswith(".jpg"):
                images.append(os.path.join(root, f))
    return images


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


def find_detection_matches_processed(dataset_root, source_folder=None):
    """Find (jpg, json) pairs where source images are in *source_folder* (or
    anywhere under *dataset_root*) and JSON outputs live in the ``_processed``
    mirror tree.

    This is the structure-agnostic replacement for ``find_detection_matches``
    and works regardless of nightly-folder layout.

    Parameters
    ----------
    dataset_root : str
        Top-level folder the user chose to process.
    source_folder : str | None
        If provided, only return matches whose source image is directly inside
        *source_folder*.  If None, all images under *dataset_root* (excluding
        the ``_processed`` sub-tree) are considered.

    Returns
    -------
    hu_matches : list[tuple[str, str]]
        (jpg_path, human_json_path) pairs where human_json_path is in _processed.
    bot_matches : list[tuple[str, str]]
        (jpg_path, bot_json_path) pairs where bot_json_path is in _processed.
    """
    # Import here to avoid circular imports
    from core.paths import get_json_output_path

    jpg_files = find_images_recursive(source_folder if source_folder is not None else dataset_root)

    hu_matches = []
    bot_matches = []
    for jpg in jpg_files:
        human_json = get_json_output_path(jpg, "", dataset_root)
        bot_json = get_json_output_path(jpg, "_botdetection", dataset_root)

        # Also check for human ground-truth JSONs placed next to source images
        human_json_source = jpg.replace(".jpg", ".json")

        if os.path.isfile(human_json):
            hu_matches.append((jpg, human_json))
        elif os.path.isfile(human_json_source):
            hu_matches.append((jpg, human_json_source))

        if os.path.isfile(bot_json):
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


def build_cuda_diagnostics():
    """Return a structured CUDA diagnostics report as text lines."""
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_build = torch.version.cuda
    cuda_backend = getattr(torch.backends, "cuda", None)
    cuda_backend_built = bool(
        cuda_backend
        and hasattr(cuda_backend, "is_built")
        and cuda_backend.is_built()
    )

    lines = [
        f"PyTorch version: {torch.__version__}",
        f"PyTorch CUDA build: {cuda_build or 'None (CPU-only build likely)'}",
        f"PyTorch CUDA backend built: {cuda_backend_built}",
        f"torch.cuda.is_available(): {cuda_available}",
        f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]

    if cuda_available:
        lines.append(f"Number of GPUs: {torch.cuda.device_count()}")
        try:
            current_device = torch.cuda.current_device()
            lines.append(f"Current device index: {current_device}")
            lines.append(f"Current GPU name: {torch.cuda.get_device_name(current_device)}")
        except Exception as exc:
            lines.append(f"Could not read current CUDA device details: {exc}")
    else:
        if not cuda_backend_built or cuda_build is None:
            lines.append("Likely cause: CPU-only torch build in this environment.")
        else:
            lines.append(
                "Likely cause: CUDA build exists, but runtime cannot access a compatible GPU/driver."
            )
    return lines


def print_device_info(selected_device=None):
    """Print CUDA diagnostics and selected runtime device."""
    selected_device = selected_device or get_device()
    print("=== Device diagnostics ===")
    for line in build_cuda_diagnostics():
        print(line)
    print(f"Selected runtime device: {selected_device}")
    if selected_device == "cpu":
        print(
            "If you expected CUDA, verify your torch install profile and the target machine's NVIDIA driver/runtime."
        )


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


# ---------------------------------------------------------------------------
# Thread-local stdout routing
#
# Replacing sys.stdout globally (the old approach) is not thread-safe: a
# second run_in_thread call overwrites the capture set by the first, causing
# output from both threads to be mixed into whichever cap was set last.
#
# Instead we install a single proxy object once.  The proxy checks a
# thread-local slot for a per-thread _OutputCapture; if one is set the write
# goes there, otherwise it falls through to the real original stdout.  This
# means each worker thread captures only its own prints with no interference.
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_real_stdout = sys.stdout   # saved before any replacement


class _RoutingWriter(io.TextIOBase):
    """sys.stdout proxy that routes writes to per-thread capture queues."""

    def write(self, s: str) -> int:
        cap = getattr(_thread_local, "capture", None)
        if cap is not None:
            return cap.write(s)
        return _real_stdout.write(s)

    def flush(self) -> None:
        cap = getattr(_thread_local, "capture", None)
        if cap is not None:
            cap.flush()
        else:
            _real_stdout.flush()

    def isatty(self) -> bool:
        return getattr(_real_stdout, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(_real_stdout, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(_real_stdout, "errors", "replace")


# Install once at import time; subsequent imports are no-ops.
if not isinstance(sys.stdout, _RoutingWriter):
    sys.stdout = _RoutingWriter()


def _log_stream_chunk(logger, chunk):
    for line in chunk.splitlines():
        if line.strip():
            logger.info(line.rstrip())


# Module-level cancel event. Set this to ask the currently-running
# run_in_thread call to stop at the next yield boundary.
_cancel_event = threading.Event()


def request_cancel():
    """Signal the currently-running pipeline step to stop."""
    _cancel_event.set()


def clear_cancel():
    """Clear any pending cancel signal (called before starting a new run)."""
    _cancel_event.clear()


def run_in_thread(fn, *args, **kwargs):
    """Run *fn* in a background thread and yield captured stdout chunks.

    On macOS, holds a ``caffeinate -s`` process for the duration so that
    screen-lock / idle power management cannot suspend the run mid-flight.

    Cancellation: call request_cancel() to stop at the next yield boundary.
    The background thread finishes its current atomic op then winds down.
    """
    clear_cancel()
    cap = _OutputCapture()
    error_holder: list = [None]
    logger = logging.getLogger("mothbot.pipeline")
    should_log_to_logger = logger.hasHandlers()

    def _worker():
        _thread_local.capture = cap   # thread-local: no race with other workers
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            _thread_local.capture = None
            cap.q.put(None)  # sentinel

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # Prevent macOS from suspending CPU-heavy background work when the screen locks.
    _caffeinate = None
    if sys.platform == "darwin":
        try:
            import subprocess as _sp
            _caffeinate = _sp.Popen(["caffeinate", "-dis"])
        except Exception:
            pass  # non-fatal if caffeinate is somehow unavailable

    try:
        while True:
            if _cancel_event.is_set():
                yield "\n⛔ Run cancelled by user.\n"
                try:
                    while True:
                        cap.q.get_nowait()
                except queue.Empty:
                    pass
                t.join(timeout=5)
                clear_cancel()
                return

            try:
                chunk = cap.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            if should_log_to_logger:
                _log_stream_chunk(logger, chunk)
            yield chunk

        t.join(timeout=10)
        if error_holder[0]:
            raise error_holder[0]
    finally:
        if _caffeinate is not None:
            _caffeinate.terminate()
            _caffeinate.wait(timeout=3)