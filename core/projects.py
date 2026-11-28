"""Projects: the folder that is processed, shared and opened as one unit.

A project folder holds its raw deployment folders and its own ``_processed`` folder
(the part people share). It is marked by ``mothbox-project.json``, written into the
project folder and, as a copy, into its ``_processed`` folder so a shared
``_processed`` still says which project it is::

    MB Projects/
      Panama_Gamboa_2026/            <- project (mothbox-project.json)
        PlotA_snappySloth/2026-07-14/*.jpg
        _processed/                  <- mothbox-project.json, dataset.json, results
      Tucson_DesertHouse/ ...

Older data has one ``_processed`` for a whole folder of projects
(``MB Projects/_processed/Panama_Gamboa_2026/...``). Folders whose results already
live there keep using it, so nothing is processed twice or split across two layouts.

``project_root_for`` decides, per image folder, which folder is the root that
``core.paths`` mirrors into ``_processed``:

1. the nearest folder at or above it holding ``mothbox-project.json``;
2. otherwise the folder whose ``_processed`` already holds its results (older data);
3. otherwise ``None``: new data whose project hasn't been chosen yet.
"""

import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone

MARKER = "mothbox-project.json"
FORMAT = "mothbox-project"
FORMAT_VERSION = 1
PROCESSED = "_processed"
_MAX_LEVELS = 24


def _ancestors(path):
    """*path* and the folders above it, nearest first."""
    path = os.path.abspath(path)
    for _ in range(_MAX_LEVELS):
        yield path
        parent = os.path.dirname(path)
        if parent == path:
            return
        path = parent


def has_marker(folder):
    return os.path.isfile(os.path.join(folder, MARKER))


def read_marker(folder):
    try:
        with open(os.path.join(folder, MARKER), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def marker_root(path):
    """The nearest folder at or above *path* that holds ``mothbox-project.json``."""
    for folder in _ancestors(path):
        if os.path.basename(folder) == PROCESSED:
            continue  # the copy inside _processed names the project; the project is one level up
        if has_marker(folder):
            return folder
    return None


def _processed_owner(path):
    """For a folder inside a ``_processed`` tree, the folder that tree belongs to."""
    parts = os.path.abspath(path).split(os.sep)
    if PROCESSED in parts:
        index = parts.index(PROCESSED)
        return os.sep.join(parts[:index]) or os.sep
    return None


def existing_results_root(collection):
    """The folder whose ``_processed`` already holds results for *collection*, if any.

    For each folder A at or above *collection* (nearest first): A itself counts when
    *collection* is A and ``A/_processed`` exists (a single night picked directly);
    otherwise A counts when ``A/_processed/<first folder below A on the way down>``
    exists. That first folder is the project or deployment, so a new night added to
    a project that was processed the older way follows that project.
    """
    collection = os.path.abspath(collection)
    for folder in _ancestors(collection):
        processed = os.path.join(folder, PROCESSED)
        if not os.path.isdir(processed):
            continue
        if folder == collection:
            return folder
        first = os.path.relpath(collection, folder).split(os.sep)[0]
        if os.path.isdir(os.path.join(processed, first)):
            return folder
    return None


def project_root_for(collection):
    """The root that ``core.paths`` mirrors *collection* into, or ``None`` if not chosen yet."""
    owner = _processed_owner(collection)
    if owner:
        return owner  # results without photos: they already sit in their project's _processed
    return marker_root(collection) or existing_results_root(collection)


def processed_mirror(collection, root):
    """Where *collection*'s results go under *root*, without creating anything."""
    if _processed_owner(collection):
        return os.path.abspath(collection)
    rel = os.path.relpath(os.path.abspath(collection), os.path.abspath(root))
    return os.path.join(os.path.abspath(root), PROCESSED, "" if rel == "." else rel)


# ──────────────────────────────────────────────────────────────
#  Choosing the project for new data
# ──────────────────────────────────────────────────────────────

def layout_options(scan_root, unresolved):
    """Ways to answer "which folder is the top folder of the project?" for the image
    folders in *unresolved* (none of which has a project yet), found by scanning
    *scan_root*.

    Returns a list of ``{"id", "label", "why", "roots", "suggested"}``; ``roots`` are
    the folders that get ``mothbox-project.json``. The suggestion follows the usual
    shape: project / deployment / night.
    """
    scan_root = os.path.abspath(scan_root)
    unresolved = [os.path.abspath(p) for p in unresolved]
    if not unresolved:
        return []
    depths = Counter(
        0 if p == scan_root else len(os.path.relpath(p, scan_root).split(os.sep)) for p in unresolved
    )
    usual_depth = depths.most_common(1)[0][0]
    name = os.path.basename(scan_root) or scan_root
    options = []

    children = sorted({
        os.path.join(scan_root, os.path.relpath(p, scan_root).split(os.sep)[0])
        for p in unresolved if p != scan_root
    })
    # Sub-folders as projects only makes sense when they hold deployments, not nights.
    if children and usual_depth >= 3 and all(p != scan_root for p in unresolved):
        n = len(children)
        options.append({
            "id": "children",
            "label": f"{name} holds several projects",
            "why": f"Each of its {n} sub-folder{'s' if n != 1 else ''} with new photos becomes its own project."
            if n != 1 else "Its one sub-folder with new photos becomes a project.",
            "roots": children,
            "suggested": usual_depth >= 3,
        })
    options.append({
        "id": "self",
        "label": f"{name} is one project",
        "why": "Everything in it is processed and shared together.",
        "roots": [scan_root],
        "suggested": usual_depth == 2 or (usual_depth >= 3 and not options),
    })
    # Picked too low (a deployment or a night): offer the folders above.
    parent = os.path.dirname(scan_root) if usual_depth < 2 else None
    for level, why in ((1, "One level up: pick this if you chose a deployment folder."),
                       (2, "Two levels up: pick this if you chose a single night.")):
        if not parent or parent == scan_root or os.path.dirname(parent) == parent:
            break
        options.append({
            "id": f"up{level}",
            "label": f"{os.path.basename(parent)} (above the folder you picked)",
            "why": why,
            "roots": [parent],
            "suggested": usual_depth == 2 - level,
        })
        scan_root, parent = parent, os.path.dirname(parent)
    if not any(o["suggested"] for o in options):
        options[0]["suggested"] = True
    return options


def write_marker(project_folder):
    """Writes ``mothbox-project.json`` into *project_folder* and its ``_processed``
    folder (creating ``_processed``). An existing marker is left as it is."""
    project_folder = os.path.abspath(project_folder)
    data = read_marker(project_folder)
    if data is None:
        data = {
            "format": FORMAT,
            "formatVersion": FORMAT_VERSION,
            "projectID": f"prj-{uuid.uuid4().hex[:12]}",
            "name": os.path.basename(project_folder),
            "processedFolder": PROCESSED,
            "createdAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        _write_json(os.path.join(project_folder, MARKER), data)
    processed = os.path.join(project_folder, PROCESSED)
    os.makedirs(processed, exist_ok=True)
    if not has_marker(processed):
        _write_json(os.path.join(processed, MARKER), data)
    return data


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
