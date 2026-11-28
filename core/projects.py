"""Projects: the folder that holds ``_processed`` is the project.

A project is processed, shared (its ``_processed`` folder) and opened as one unit.
Inside it, people organize their deployment and night folders however they like::

    MB Projects/                     <- a folder of projects
      Tucson/                        <- project (mothbox-project.json, _processed/)
        DesertHouse/Cactus/<night>/*.jpg
        _processed/                  <- results, shared as one
      Maine/ ...

The project folder is marked by ``mothbox-project.json`` (a copy sits inside
``_processed``, so a shared ``_processed`` still names its project). Process writes
it when it first creates a project's ``_processed``.

``project_root_for`` decides, per image folder, which project it belongs to: the
nearest folder at or above it holding ``mothbox-project.json`` or ``_processed``.
A folder inside a project belongs to that project. New photos outside any project
get the folder that was picked (``default_project_for_scan``), unless that folder
holds projects already, in which case Setup asks.
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
    """The project *collection* belongs to (the root ``core.paths`` mirrors it into),
    or ``None`` when it isn't in any project yet."""
    owner = _processed_owner(collection)
    if owner:
        return owner  # results without photos: they already sit in their project's _processed
    for folder in _ancestors(collection):
        if has_marker(folder) or is_project_processed(os.path.join(folder, PROCESSED)):
            return folder
    return None


def is_project_processed(processed):
    """True if *processed* is a project's ``_processed``: it has the marker, Classify's
    records, or results. One left holding only backups (after its folders were split
    into projects of their own) no longer makes a project."""
    if not os.path.isdir(processed):
        return False
    if has_marker(processed) or os.path.isfile(os.path.join(processed, "dataset.json")):
        return True
    try:
        return any(e.is_dir() and not e.name.startswith(("_", ".")) for e in os.scandir(processed))
    except OSError:
        return False


def holds_projects(folder):
    """True if a project sits at or below *folder* (a ``_processed`` folder or marker)."""
    folder = os.path.abspath(folder)
    if has_marker(folder):
        return True
    for root, dirs, files in os.walk(folder):
        if MARKER in files or (PROCESSED in dirs and is_project_processed(os.path.join(root, PROCESSED))):
            return True
        dirs[:] = [d for d in dirs if d != PROCESSED and d.lower() != "patches"]
    return False


def default_project_for_scan(scan_root, unresolved):
    """The project new photos found by scanning *scan_root* go into without asking:
    the picked folder itself, unless it already holds projects (then ``None``: ask)."""
    if not unresolved or holds_projects(scan_root):
        return None
    return os.path.abspath(scan_root)


def new_project_options(scan_root, unresolved):
    """For a picked folder that holds projects and also new photos outside them: ways
    to place the new photos. Same shape as ``layout_options``."""
    scan_root = os.path.abspath(scan_root)
    tops = sorted({
        os.path.join(scan_root, os.path.relpath(os.path.abspath(p), scan_root).split(os.sep)[0])
        for p in unresolved if os.path.abspath(p) != scan_root
    })
    name = os.path.basename(scan_root) or scan_root
    options = []
    if tops:
        names = ", ".join(os.path.basename(t) for t in tops[:4]) + (" …" if len(tops) > 4 else "")
        options.append({
            "id": "children",
            "label": f"Each new folder is its own project ({names})",
            "why": "Next to the projects already in " + name + ".",
            "roots": tops,
            "suggested": True,
        })
    options.append({
        "id": "self",
        "label": f"{name} is the project for the new photos",
        "why": "Its own _processed holds them; the projects inside it stay as they are.",
        "roots": [scan_root],
        "suggested": not tops,
    })
    return options


def processed_mirror(collection, root):
    """Where *collection*'s results go under *root*, without creating anything."""
    if _processed_owner(collection):
        return os.path.abspath(collection)
    from core.deployments import processed_rel_for
    rel = os.path.relpath(os.path.abspath(collection), os.path.abspath(root))
    rel = "" if rel == "." else processed_rel_for(root, rel)
    return os.path.join(os.path.abspath(root), PROCESSED, *(rel.split("/") if rel else []))


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
    os.makedirs(project_folder, exist_ok=True)
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


# ──────────────────────────────────────────────────────────────
#  Moving older data to the per-project layout
# ──────────────────────────────────────────────────────────────
#
# Older Process runs kept one _processed for a whole folder of projects:
#     MB Projects/_processed/DesertHouse/{dataset.json, 02_records/, Cactus/…}
# The per-project layout keeps it inside the project:
#     MB Projects/DesertHouse/_processed/{dataset.json, 02_records/, Cactus/…}
# Every path in the package records is relative to the package folder, so moving
# the folder (one rename, on the same drive) keeps human IDs, sessions and patches
# working. Nothing is copied or deleted.


def _has_detections_two_levels_down(folder):
    """True if *folder*/<deployment>/<night>/ holds a ``*_botdetection.json``."""
    try:
        for deployment in os.scandir(folder):
            if not deployment.is_dir() or deployment.name.startswith("."):
                continue
            for night in os.scandir(deployment.path):
                if not night.is_dir() or night.name.startswith("."):
                    continue
                for entry in os.scandir(night.path):
                    if entry.name.endswith("_botdetection.json"):
                        return True
    except OSError:
        return False
    return False


def _is_project_results(folder):
    """A folder inside an older shared _processed that holds one project's results."""
    return os.path.isfile(os.path.join(folder, "dataset.json")) or _has_detections_two_levels_down(folder)


def _processed_folders_under(directory):
    found = []
    for root, dirs, _files in os.walk(os.path.abspath(directory)):
        if PROCESSED in dirs:
            found.append(os.path.join(root, PROCESSED))
        dirs[:] = sorted(d for d in dirs if d != PROCESSED and d.lower() != "patches")
    return found


def older_layout_moves(scan_root, collections=()):
    """Projects found from *scan_root* whose results are still in an older shared
    ``_processed``, each as ``{"name", "source", "project", "destination",
    "has_photos", "has_package", "problem"}``. ``problem`` is set (and the move
    skipped) when the destination is already taken.

    Looks in every ``_processed`` at or below *scan_root*, and in the one above it
    that holds results for *collections* (when a single project was picked).
    """
    scan_root = os.path.abspath(scan_root)
    shared = set(_processed_folders_under(scan_root))
    for collection in collections:
        root = existing_results_root(collection)
        if root and not marker_root(collection):
            shared.add(os.path.join(root, PROCESSED))

    moves = []
    for processed in sorted(shared):
        owner = os.path.dirname(processed)
        # Already a project's own _processed: nothing to move.
        if has_marker(owner) or has_marker(processed) or os.path.isfile(os.path.join(processed, "dataset.json")):
            continue
        # Above the picked folder: only the project on the way down to it.
        only = os.path.relpath(scan_root, owner).split(os.sep)[0] if scan_root.startswith(owner + os.sep) else None
        try:
            entries = sorted(os.scandir(processed), key=lambda e: e.name.lower())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or (only and entry.name != only):
                continue
            if not _is_project_results(entry.path):
                continue
            project = os.path.join(owner, entry.name)
            destination = os.path.join(project, PROCESSED)
            problem = None
            if os.path.exists(destination):
                problem = f"{entry.name} already has its own _processed folder"
            elif os.path.exists(project) and not os.path.isdir(project):
                problem = f"a file named {entry.name} is in the way"
            moves.append({
                "name": entry.name,
                "source": entry.path,
                "project": project,
                "destination": destination,
                "has_photos": os.path.isdir(project),
                "has_package": os.path.isfile(os.path.join(entry.path, "dataset.json")),
                "problem": problem,
            })
    return moves


def apply_older_layout_move(move):
    """Moves one project's results into ``<project>/_processed`` and marks the project.

    A rename only: it fails (and changes nothing) rather than copying if the
    destination is on another drive or already exists.
    """
    source, project, destination = move["source"], move["project"], move["destination"]
    if not os.path.isdir(source):
        raise FileNotFoundError(f"{source} is no longer there")
    if os.path.exists(destination):
        raise FileExistsError(f"{destination} already exists")
    os.makedirs(project, exist_ok=True)
    os.rename(source, destination)
    write_marker(project)
    # Deployment links written while the results were shared still include the
    # project's folder name; inside the project they are relative to it.
    from core.deployments import clear_cache, read_deployment, write_deployment
    prefix = move["name"] + "/"
    try:
        for entry in os.scandir(destination):
            data = read_deployment(entry.path) if entry.is_dir() else None
            raw_folder = (data or {}).get("rawFolder")
            if isinstance(raw_folder, str) and raw_folder.startswith(prefix):
                data["rawFolder"] = raw_folder[len(prefix):]
                write_deployment(entry.path, data)
    except OSError:
        pass
    clear_cache()
    # The shared _processed is left behind once empty; remove only an empty folder.
    old = os.path.dirname(source)
    try:
        if not os.listdir(old):
            os.rmdir(old)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────
#  Moving a deployment to another project
# ──────────────────────────────────────────────────────────────

def project_id(project_folder):
    return (read_marker(project_folder) or {}).get("projectID")


def deployment_parts(project_folder, deployment):
    """(raw folder, processed folder) of *deployment* (its path inside the project,
    e.g. ``DesertHouse/Cactus``). Either may not exist: photos not copied over, or
    not processed yet."""
    from core.deployments import processed_rel_for
    raw = os.path.join(project_folder, *deployment.split("/"))
    processed_rel = processed_rel_for(project_folder, deployment)
    processed = os.path.join(project_folder, PROCESSED, *processed_rel.split("/"))
    return raw, processed


def human_ids_for_deployment(project_folder, folder):
    """How many human identifications Classify holds, in this project's records, for
    patches inside *folder* (its path inside the project, e.g. ``DesertHouse/Cactus``).
    They stay in those files when the folder moves, until Classify moves them across.

    The records are in ``_processed`` itself, or (older setups) in a package inside it
    named after the first folder, whose paths start below that folder."""
    parts = folder.replace(os.sep, "/").strip("/").split("/")
    processed = os.path.join(project_folder, PROCESSED)
    candidates = [(processed, "/".join(parts))]
    if os.path.isfile(os.path.join(processed, parts[0], "dataset.json")):
        candidates.append((os.path.join(processed, parts[0]), "/".join(parts[1:])))
    return sum(_human_ids_in_package(package, prefix) for package, prefix in candidates)


def _ndjson_rows(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _human_ids_in_package(package, prefix):
    """Human identification rows in *package* for patches whose path starts with *prefix*."""
    if not os.path.isfile(os.path.join(package, "dataset.json")):
        return 0
    start = prefix + "/" if prefix else ""
    patch_ids = {
        row.get("patchID") or row.get("patch_id")
        for row in _ndjson_rows(os.path.join(package, "02_records", "patches.ndjson"))
        if str(row.get("assetPath") or row.get("asset_path") or "").startswith(start)
    }
    if not patch_ids:
        return 0
    folder = os.path.join(package, "03_classifications")
    try:
        names = [n for n in os.listdir(folder) if n.endswith(".ndjson") and not n.startswith("_")]
    except OSError:
        return 0
    return sum(
        1
        for name in names
        for row in _ndjson_rows(os.path.join(folder, name))
        if (row.get("patchID") or row.get("patch_id")) in patch_ids
    )


def plan_deployment_move(source_project, deployment, destination_project):
    """What moving *deployment* from *source_project* to *destination_project* does:
    ``{"moves": [(from, to), …], "problem": str | None, "new_project": bool}``."""
    raw, processed = deployment_parts(source_project, deployment)
    destination_project = os.path.abspath(destination_project)
    is_project = has_marker(destination_project) or os.path.isdir(os.path.join(destination_project, PROCESSED))
    plan = {"moves": [], "problem": None, "new_project": not is_project, "processed_destination": None}
    if os.path.abspath(source_project) == destination_project:
        plan["problem"] = "That is the project it is already in."
        return plan
    # It keeps its place inside the project: DesertHouse/Cactus stays DesertHouse/Cactus.
    if os.path.isdir(raw):
        plan["moves"].append((raw, os.path.join(destination_project, *deployment.split("/"))))
    if os.path.isdir(processed):
        plan["processed_destination"] = os.path.join(destination_project, PROCESSED, *deployment.split("/"))
        plan["moves"].append((processed, plan["processed_destination"]))
    if not plan["moves"]:
        plan["problem"] = f"{deployment} is no longer in {os.path.basename(source_project)}."
    for _src, dst in plan["moves"]:
        if os.path.exists(dst):
            plan["problem"] = f"{os.path.relpath(dst, os.path.dirname(destination_project))} already exists."
    if os.path.exists(destination_project) and not os.path.isdir(destination_project):
        plan["problem"] = f"A file named {os.path.basename(destination_project)} is in the way."
    return plan


def move_deployment(source_project, deployment, destination_project):
    """Moves a deployment's photos and results to another project (made if needed),
    with renames only, and records the move in its deployment.json. Human IDs stay in
    the source project's Classify records; Classify brings them across."""
    from core.deployments import clear_cache, record_move
    plan = plan_deployment_move(source_project, deployment, destination_project)
    if plan["problem"]:
        raise ValueError(plan["problem"])
    if plan["new_project"]:
        write_marker(destination_project)  # makes the project and its _processed
    # An existing project is left unmarked here: its Classify records are combined
    # first (Mothbot Classify), so no marker is added to an older _processed yet.
    done = []
    try:
        for src, dst in plan["moves"]:
            os.makedirs(os.path.dirname(dst), exist_ok=True)  # e.g. DesertHouse/ in the new project
            os.rename(src, dst)
            done.append((src, dst))
    except OSError:
        for src, dst in reversed(done):  # put back what already moved
            os.rename(dst, src)
        raise
    finally:
        clear_cache()
    processed_dst = plan["processed_destination"]
    if processed_dst and _has_classify_records(source_project):
        # Classify moves this deployment's records and identifications across.
        source_processed = next(src for src, dst in plan["moves"] if dst == processed_dst)
        note_records_to_move(
            destination_project,
            source_project,
            os.path.relpath(source_processed, os.path.join(source_project, PROCESSED)),
            os.path.relpath(processed_dst, os.path.join(destination_project, PROCESSED)),
        )
    if processed_dst:
        record_move(
            processed_dst,
            deployment,
            project_id(destination_project),
            {
                "projectID": project_id(source_project),
                "projectName": os.path.basename(os.path.abspath(source_project)),
                "classifyRecords": os.path.isfile(os.path.join(source_project, PROCESSED, "dataset.json")),
            },
        )
    return plan


# ──────────────────────────────────────────────────────────────
#  Merging and splitting projects
# ──────────────────────────────────────────────────────────────
#
# A project is the folder holding _processed, so changing which folder that is changes
# the projects: a higher folder merges the projects below it, lower folders split one.
# Process moves the results (renames only); Classify keeps the records and human
# identifications in step:
#   - merged projects' _processed folders land in <new project>/_processed/<folder>/,
#     the per-folder layout Classify combines into one set of records on open;
#   - split-off (and moved) folders get a note, RECORDS_TO_MOVE, in their new
#     _processed naming the project whose records they were in. Classify moves those
#     records across when it next sees both projects.

RECORDS_TO_MOVE = "mothbox-records-to-move.json"


def note_records_to_move(receiver_project, giver_project, from_folder, to_folder):
    """Adds to *receiver_project*'s note: the results of *from_folder* (inside the
    giver's _processed) now sit at *to_folder* (inside the receiver's _processed)."""
    from datetime import datetime as _dt
    processed = os.path.join(receiver_project, PROCESSED)
    os.makedirs(processed, exist_ok=True)
    path = os.path.join(processed, RECORDS_TO_MOVE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict) or not isinstance(data.get("moves"), list):
        data = {"format": "mothbox-records-to-move", "formatVersion": 1, "moves": []}
    data["moves"].append({
        "fromProject": os.path.basename(os.path.abspath(giver_project)),
        "fromProjectID": project_id(giver_project),
        # Where the giver is, from the receiver's project folder ("..", "../Tucson").
        "fromProjectPath": os.path.relpath(os.path.abspath(giver_project), os.path.abspath(receiver_project)).replace(os.sep, "/"),
        "fromFolder": from_folder.replace(os.sep, "/").strip("/"),
        "toFolder": to_folder.replace(os.sep, "/").strip("/"),
        "at": _dt.now(timezone.utc).replace(microsecond=0).isoformat(),
    })
    _write_json(path, data)


def _has_classify_records(project_folder):
    return os.path.isfile(os.path.join(project_folder, PROCESSED, "dataset.json"))


def _rewrite_links(processed_folder, change):
    """Applies *change* to every deployment.json rawFolder within two levels."""
    from core.deployments import read_deployment, write_deployment

    def visit(folder, depth):
        try:
            entries = [e for e in os.scandir(folder) if e.is_dir() and not e.name.startswith((".", "_"))]
        except OSError:
            return
        for entry in entries:
            data = read_deployment(entry.path)
            raw_folder = (data or {}).get("rawFolder")
            if isinstance(raw_folder, str):
                new = change(raw_folder)
                if new != raw_folder:
                    data["rawFolder"] = new
                    write_deployment(entry.path, data)
            elif depth < 3:
                visit(entry.path, depth + 1)

    visit(processed_folder, 1)


def split_plan(project_folder, folders=None):
    """Splitting *project_folder* into projects made of its sub-folders (all of them, or
    *folders*): ``{"moves": [{"folder", "source", "destination", "has_results"}], "problem"}``."""
    from core.deployments import processed_rel_for
    project_folder = os.path.abspath(project_folder)
    if folders is None:
        try:
            folders = sorted(e.name for e in os.scandir(project_folder)
                             if e.is_dir() and not e.name.startswith((".", "_")))
        except OSError:
            folders = []
    plan = {"moves": [], "problem": None}
    for name in folders:
        child = os.path.join(project_folder, name)
        if not os.path.isdir(child):
            continue
        processed_rel = processed_rel_for(project_folder, name)
        source = os.path.join(project_folder, PROCESSED, *processed_rel.split("/"))
        destination = os.path.join(child, PROCESSED)
        if os.path.exists(destination):
            plan["problem"] = f"{name} already has its own _processed folder."
        plan["moves"].append({"folder": name, "source": source, "destination": destination,
                              "has_results": os.path.isdir(source)})
    if not plan["moves"]:
        plan["problem"] = f"{os.path.basename(project_folder)} has no folders to split into projects."
    return plan


def split_project(project_folder, folders=None):
    """Makes each sub-folder of *project_folder* (all, or *folders*) its own project."""
    from core.deployments import clear_cache
    plan = split_plan(project_folder, folders)
    if plan["problem"]:
        raise ValueError(plan["problem"])
    records = _has_classify_records(project_folder)
    done = []
    try:
        for move in plan["moves"]:
            if move["has_results"]:
                os.rename(move["source"], move["destination"])
                done.append(move)
    except OSError:
        for move in reversed(done):
            os.rename(move["destination"], move["source"])
        clear_cache()
        raise
    for move in plan["moves"]:
        child = os.path.join(project_folder, move["folder"])
        write_marker(child)
        prefix = move["folder"] + "/"
        _rewrite_links(move["destination"], lambda raw: raw[len(prefix):] if raw.startswith(prefix) else raw)
        if records and move["has_results"]:
            from_folder = os.path.relpath(move["source"], os.path.join(project_folder, PROCESSED))
            note_records_to_move(child, project_folder, from_folder, "")
    # With nothing left in it, the old project is now just a folder of projects.
    processed = os.path.join(project_folder, PROCESSED)
    if not any(e.is_dir() and not e.name.startswith(("_", ".")) for e in os.scandir(processed)):
        for marker in (os.path.join(project_folder, MARKER), os.path.join(processed, MARKER)):
            if os.path.isfile(marker):
                os.remove(marker)
        try:
            if not os.listdir(processed):
                os.rmdir(processed)
        except OSError:
            pass
    clear_cache()
    return plan


def merge_plan(folder):
    """Making *folder* one project of the projects inside it:
    ``{"moves": [{"project", "source", "destination"}], "problem"}``."""
    folder = os.path.abspath(folder)
    plan = {"moves": [], "problem": None}
    try:
        children = sorted(e.path for e in os.scandir(folder) if e.is_dir() and not e.name.startswith((".", "_")))
    except OSError:
        children = []
    for child in children:
        processed = os.path.join(child, PROCESSED)
        if not (has_marker(child) or is_project_processed(processed)):
            continue
        destination = os.path.join(folder, PROCESSED, os.path.basename(child))
        if os.path.exists(destination):
            plan["problem"] = f"{os.path.relpath(destination, os.path.dirname(folder))} already exists."
        plan["moves"].append({"project": os.path.basename(child), "source": processed, "destination": destination,
                              "has_results": os.path.isdir(processed)})
    if len(plan["moves"]) < 1:
        plan["problem"] = f"There are no projects inside {os.path.basename(folder)} to merge."
    return plan


def merge_projects(folder):
    """Makes *folder* one project holding the projects inside it."""
    from core.deployments import clear_cache
    plan = merge_plan(folder)
    if plan["problem"]:
        raise ValueError(plan["problem"])
    os.makedirs(os.path.join(folder, PROCESSED), exist_ok=True)
    done = []
    try:
        for move in plan["moves"]:
            if move["has_results"]:
                os.rename(move["source"], move["destination"])
                done.append(move)
    except OSError:
        for move in reversed(done):
            os.rename(move["destination"], move["source"])
        clear_cache()
        raise
    for move in plan["moves"]:
        child = os.path.join(folder, move["project"])
        # The merged projects are folders of the new one now.
        for marker in (os.path.join(child, MARKER), os.path.join(move["destination"], MARKER)):
            if os.path.isfile(marker):
                os.remove(marker)
        prefix = move["project"] + "/"
        _rewrite_links(move["destination"], lambda raw: raw if raw.startswith(prefix) else prefix + raw)
    write_marker(folder)
    clear_cache()
    return plan
