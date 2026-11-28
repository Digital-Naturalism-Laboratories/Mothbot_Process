"""Deployment links: ``deployment.json`` in each processed deployment folder.

A project keeps its raw deployment folders and their results in ``_processed``::

    Desert2026/Cactus/<night>/*.jpg                          raw photos
    Desert2026/_processed/Cactus/deployment.json             this file
    Desert2026/_processed/Cactus/<night>/*_botdetection.json results

``deployment.json`` names the deployment and where its photos are, so the two
halves are recognized after one of them is renamed or moved::

    {"format": "mothbox-deployment", "formatVersion": 1,
     "deploymentID": "Cactus", "projectID": "prj-…",
     "rawFolder": "Cactus",              # relative to the folder that holds _processed
     "movedFrom": [{"projectID": "prj-…", "projectName": "…", "at": "…"}]}

``processed_rel_for`` is what ``core.paths`` uses to put a raw folder's results in
the right place: a renamed processed deployment keeps receiving its outputs, and a
renamed raw deployment finds its earlier results instead of starting a second copy.
"""

import json
import os
from datetime import datetime, timezone

DEPLOYMENT_FILE = "deployment.json"
FORMAT = "mothbox-deployment"
FORMAT_VERSION = 1
PROCESSED = "_processed"
_LINK_DEPTH = 2  # deployment.json sits 1 (project layout) or 2 (older layout) levels into _processed

_links_cache = {}


def clear_cache():
    """Forget the links read from disk (call after folders were moved or renamed)."""
    _links_cache.clear()


def read_deployment(processed_deployment):
    try:
        with open(os.path.join(processed_deployment, DEPLOYMENT_FILE), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_deployment(processed_deployment, data):
    os.makedirs(processed_deployment, exist_ok=True)
    path = os.path.join(processed_deployment, DEPLOYMENT_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _rel(path):
    return path.replace(os.sep, "/").strip("/")


def ensure_deployment_file(processed_deployment, root, raw_deployment, project_id=None):
    """Writes ``deployment.json`` for a deployment when it has none yet, or fills in
    what an older one lacks. Never changes a deployment ID that is already there."""
    data = read_deployment(processed_deployment) or {}
    changed = False
    defaults = {
        "format": FORMAT,
        "formatVersion": FORMAT_VERSION,
        "deploymentID": os.path.basename(os.path.normpath(raw_deployment)),
        "rawFolder": _rel(os.path.relpath(raw_deployment, root)),
    }
    if project_id:
        defaults["projectID"] = project_id
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
            changed = True
    if changed:
        write_deployment(processed_deployment, data)
    return data


def _scan_links(root):
    """``{raw rel path: processed rel path}`` for every deployment.json under root/_processed."""
    links = {}
    processed = os.path.join(root, PROCESSED)
    if not os.path.isdir(processed):
        return links

    def visit(folder, depth):
        if depth > _LINK_DEPTH:
            return
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or entry.name.lower() == "patches":
                continue
            data = read_deployment(entry.path)
            if data and isinstance(data.get("rawFolder"), str) and data["rawFolder"].strip("/"):
                links[_rel(data["rawFolder"])] = _rel(os.path.relpath(entry.path, processed))
            elif depth < _LINK_DEPTH:
                visit(entry.path, depth + 1)

    visit(processed, 1)
    return links


def links_for_root(root):
    root = os.path.realpath(root)
    if root not in _links_cache:
        _links_cache[root] = _scan_links(root)
    return _links_cache[root]


def _photo_stems(raw_folder, limit=3):
    stems = []
    try:
        for entry in sorted(os.scandir(raw_folder), key=lambda e: e.name):
            if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg")):
                stems.append(os.path.splitext(entry.name)[0])
                if len(stems) >= limit:
                    break
    except OSError:
        pass
    return stems


def _find_renamed_raw(root, rel_parts, links):
    """A raw deployment renamed after it was processed: its old results sit in a
    processed deployment whose rawFolder no longer exists. Matches by night folder
    and photo name, then updates that deployment.json. Returns (raw rel, processed rel)."""
    if len(rel_parts) < 2:
        return None
    for depth in range(1, len(rel_parts)):
        raw_prefix = "/".join(rel_parts[:depth])
        rest = rel_parts[depth:]
        night_raw = os.path.join(root, *rel_parts)
        stems = _photo_stems(night_raw) if rest else []
        if not stems:
            continue
        for old_raw, processed_rel in list(links.items()):
            if os.path.isdir(os.path.join(root, old_raw)):
                continue  # still has its photos: not the one that was renamed
            candidate = os.path.join(root, PROCESSED, processed_rel, *rest)
            if any(os.path.isfile(os.path.join(candidate, stem + "_botdetection.json")) for stem in stems):
                processed_deployment = os.path.join(root, PROCESSED, processed_rel)
                data = read_deployment(processed_deployment) or {}
                data["rawFolder"] = raw_prefix
                write_deployment(processed_deployment, data)
                del links[old_raw]
                links[raw_prefix] = processed_rel
                return raw_prefix, processed_rel
    return None


def processed_rel_for(root, rel):
    """Where results for the raw folder at *rel* (relative to *root*) go inside
    ``root/_processed``: through a deployment link when one applies, else *rel*."""
    rel = _rel(rel)
    if rel in ("", "."):
        return rel
    links = links_for_root(root)
    parts = rel.split("/")
    for depth in range(len(parts), 0, -1):
        prefix = "/".join(parts[:depth])
        if prefix in links:
            return "/".join([links[prefix]] + parts[depth:])
    # No link: use the plain mirror, unless that has nothing yet and the folder
    # was renamed after it was processed.
    if not os.path.exists(os.path.join(os.path.realpath(root), PROCESSED, *parts[:1])):
        found = _find_renamed_raw(os.path.realpath(root), parts, links)
        if found:
            raw_prefix, processed_rel = found
            return "/".join([processed_rel] + parts[len(raw_prefix.split("/")):])
    return rel


def raw_rel_for(root, processed_rel):
    """The raw folder (relative to *root*) whose results are at *processed_rel*."""
    processed_rel = _rel(processed_rel)
    reverse = {p: r for r, p in links_for_root(root).items()}
    parts = processed_rel.split("/")
    for depth in range(len(parts), 0, -1):
        prefix = "/".join(parts[:depth])
        if prefix in reverse:
            return "/".join([reverse[prefix]] + parts[depth:])
    return processed_rel


def record_move(processed_deployment, raw_folder_rel, project_id, moved_from):
    """Updates ``deployment.json`` after a deployment moved to another project."""
    data = read_deployment(processed_deployment) or {
        "format": FORMAT,
        "formatVersion": FORMAT_VERSION,
        "deploymentID": os.path.basename(os.path.normpath(processed_deployment)),
    }
    data["rawFolder"] = _rel(raw_folder_rel)
    if project_id:
        data["projectID"] = project_id
    history = data.get("movedFrom") if isinstance(data.get("movedFrom"), list) else []
    history.append({**moved_from, "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()})
    data["movedFrom"] = history
    write_deployment(processed_deployment, data)
    return data
