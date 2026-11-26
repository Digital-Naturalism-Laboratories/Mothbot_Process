"""Session files (Automated Insect Monitoring Ontology: Session, Capture Window).

Each deployment gets one ``_processed/<deployment>/sessions.json`` that writes its
sessions down explicitly, so tools that import the data (Mothbot Classify, Antenna)
keep them as they are instead of regrouping the photos by their own rules:

    {
      "format": "mothbox-sessions",
      "formatVersion": 1,
      "deploymentID": "Deployment_A",
      "sessions": [
        {
          "sessionID": "Deployment_A__2026-07-27",
          "start": "2026-07-27T12:00:00-05:00",
          "end": "2026-07-28T11:59:59-05:00",
          "definedBy": "nightly-noon-to-noon",
          "captureWindows": [
            {"captureWindowID": "Deployment_A__2026-07-27__w1",
             "start": "2026-07-27T19:00:00-05:00", "end": "2026-07-27T19:59:00-05:00",
             "captureIntervalSeconds": 60, "captureCount": 60}
          ]
        }
      ]
    }

- The default session is one night: noon to 11:59:59 the next morning, local time.
- Sessions are matched to captures by time (from the photo file name), not by folder,
  so photos taken after midnight land in the right night.
- A capture window is a run of captures without a long gap; a gap of more than three
  times the usual interval (and at least five minutes) starts a new window.
- If the file exists, sessions are only ever added for new nights. Existing sessions,
  including any a person edited, are never changed.

Field names are camelCase, as for all files (ontology: Field Names).
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median

SESSIONS_FILE = "sessions.json"
FORMAT = "mothbox-sessions"
FORMAT_VERSION = 1
DEFAULT_RULE = "nightly-noon-to-noon"

_PATTERN_A = re.compile(r"(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})")  # 2025_06_22__04_58_06
_PATTERN_B = re.compile(r"(?<!\d)(\d{14})(?!\d)")  # 20250622045806
_PATTERN_C = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})([+-]\d{2})-(\d{2})")  # 2025-06-22T04-58-06-05-00


def capture_time_from_filename(name):
    """``(local datetime, "+HH:MM" offset or None)`` from a Mothbox photo name, or ``(None, None)``."""
    base = os.path.basename(str(name))
    m = _PATTERN_C.search(base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S"), f"{m.group(2)}:{m.group(3)}"
        except ValueError:
            pass
    m = _PATTERN_A.search(base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y_%m_%d__%H_%M_%S"), None
        except ValueError:
            pass
    m = _PATTERN_B.search(base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), None
        except ValueError:
            pass
    return None, None


def night_of(local_time):
    """The night a capture belongs to: its date, or the day before if it was taken before noon."""
    return local_time.date() if local_time.hour >= 12 else local_time.date() - timedelta(days=1)


def _iso(local_time, offset):
    text = local_time.strftime("%Y-%m-%dT%H:%M:%S")
    return text + offset if offset else text


def capture_windows(times, min_gap=timedelta(minutes=5), gap_factor=3.0):
    """Splits sorted capture times into runs: ``[(start, end, interval_seconds, count), ...]``."""
    times = sorted(times)
    if not times:
        return []
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if b > a]
    usual = median(gaps) if gaps else 0
    split_after = max(min_gap.total_seconds(), gap_factor * usual)

    windows = []
    run = [times[0]]
    for prev, current in zip(times, times[1:]):
        if (current - prev).total_seconds() > split_after:
            windows.append(run)
            run = []
        run.append(current)
    windows.append(run)

    out = []
    for run in windows:
        run_gaps = [(b - a).total_seconds() for a, b in zip(run, run[1:]) if b > a]
        interval = round(median(run_gaps)) if run_gaps else None
        out.append((run[0], run[-1], interval, len(run)))
    return out


def build_default_sessions(deployment_id, photo_names):
    """Nightly sessions, with capture windows, for the photos whose names carry a time."""
    by_night = defaultdict(list)
    offsets = {}
    for name in photo_names:
        local_time, offset = capture_time_from_filename(name)
        if local_time is None:
            continue
        night = night_of(local_time)
        by_night[night].append(local_time)
        if offset and night not in offsets:
            offsets[night] = offset

    sessions = []
    for night in sorted(by_night):
        offset = offsets.get(night)
        session_id = f"{deployment_id}__{night.isoformat()}"
        start = datetime(night.year, night.month, night.day, 12, 0, 0)
        end = start + timedelta(days=1) - timedelta(seconds=1)
        windows = []
        for index, (w_start, w_end, interval, count) in enumerate(capture_windows(by_night[night]), start=1):
            window = {
                "captureWindowID": f"{session_id}__w{index}",
                "start": _iso(w_start, offset),
                "end": _iso(w_end, offset),
                "captureCount": count,
            }
            if interval is not None:
                window["captureIntervalSeconds"] = interval
            windows.append(window)
        sessions.append({
            "sessionID": session_id,
            "start": _iso(start, offset),
            "end": _iso(end, offset),
            "definedBy": DEFAULT_RULE,
            "captureWindows": windows,
        })
    return sessions


def _wall_clock(text):
    """Local wall-clock time from an ISO time, ignoring any UTC offset; None if unreadable."""
    try:
        return datetime.strptime(str(text)[:19], "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def _span(session):
    start, end = _wall_clock(session.get("start")), _wall_clock(session.get("end"))
    return (start, end) if start and end and start <= end else None


def _overlaps(a, b):
    return bool(a and b) and a[0] <= b[1] and b[0] <= a[1]


def load_sessions_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and isinstance(data.get("sessions"), list) else None
    except (OSError, ValueError):
        return None


def update_sessions_file(deployment_dir, deployment_id, photo_names):
    """Adds default sessions for new nights to ``<deployment_dir>/sessions.json``.

    Existing sessions are kept exactly as they are. Returns the number of sessions added.
    """
    path = os.path.join(deployment_dir, SESSIONS_FILE)
    existing = load_sessions_file(path)
    if existing is None and os.path.exists(path):
        # Unreadable file: leave it for a person to look at rather than overwrite it.
        print(f"⚠️  {path} could not be read; not adding sessions.")
        return 0

    data = existing or {"format": FORMAT, "formatVersion": FORMAT_VERSION, "deploymentID": deployment_id, "sessions": []}
    current = [s for s in data["sessions"] if isinstance(s, dict)]
    known = {s.get("sessionID") for s in current}
    # A night already covered by a session (one a person edited or renamed, say) gets no
    # second, overlapping default session: captures are matched to sessions by time.
    spans = [span for span in (_span(s) for s in current) if span]
    added = [
        s for s in build_default_sessions(deployment_id, photo_names)
        if s["sessionID"] not in known and not any(_overlaps(_span(s), other) for other in spans)
    ]
    if not added:
        return 0

    data["sessions"] = data["sessions"] + added
    data["sessions"].sort(key=lambda s: str(s.get("start", "")) if isinstance(s, dict) else "")
    os.makedirs(deployment_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return len(added)


def write_default_sessions(image_paths, dataset_root):
    """Groups photos by deployment and adds their default sessions. Night folders are
    expected inside deployment folders (``<deployment>/<night>/<photo>.jpg``)."""
    from core.paths import get_processed_folder

    by_deployment = defaultdict(list)
    for image_path in image_paths:
        night_folder = os.path.dirname(os.path.abspath(image_path))
        deployment_folder = os.path.dirname(night_folder)
        by_deployment[deployment_folder].append(os.path.basename(image_path))

    total = 0
    for deployment_folder, names in by_deployment.items():
        try:
            processed_dir = get_processed_folder(deployment_folder, dataset_root)
        except ValueError:
            # The chosen folder is a single night (or photos without a night folder),
            # so there is no deployment folder inside it to hold the sessions file.
            print(f"Sessions not written for {deployment_folder}: choose the deployment or dataset folder to get a sessions file.")
            continue
        deployment_id = os.path.basename(os.path.normpath(deployment_folder))
        added = update_sessions_file(processed_dir, deployment_id, names)
        if added:
            print(f"Added {added} session(s) to {os.path.join(processed_dir, SESSIONS_FILE)}")
        total += added
    return total
