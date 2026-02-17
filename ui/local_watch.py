#!/usr/bin/env python3
"""Local dev watcher that auto-restarts Gradio UI on code changes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


WATCH_DIRS = ("ui", "pipeline", "core")
WATCH_EXTENSIONS = (".py", ".toml", ".yaml", ".yml")
POLL_INTERVAL_SECONDS = 1.0


def _snapshot_mtimes(root: Path) -> dict[str, float]:
    mtimes: dict[str, float] = {}
    for rel_dir in WATCH_DIRS:
        base = root / rel_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in WATCH_EXTENSIONS:
                continue
            try:
                mtimes[str(path)] = path.stat().st_mtime
            except FileNotFoundError:
                # File may disappear between discovery and stat().
                continue
    return mtimes


def _has_changed(previous: dict[str, float], current: dict[str, float]) -> bool:
    if previous.keys() != current.keys():
        return True
    for file_path, old_mtime in previous.items():
        if current.get(file_path) != old_mtime:
            return True
    return False


def _start_ui_process() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "ui.local_main"], cwd=Path(__file__).resolve().parent.parent)


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    print("Watching for changes in ui/, pipeline/, core/ ...")
    print("Press Ctrl+C to stop.\n")

    child = _start_ui_process()
    previous_snapshot = _snapshot_mtimes(root)

    try:
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            current_snapshot = _snapshot_mtimes(root)
            if _has_changed(previous_snapshot, current_snapshot):
                print("\nCode change detected. Restarting UI...\n")
                _stop_process(child)
                child = _start_ui_process()
                previous_snapshot = current_snapshot

            if child.poll() is not None:
                # If UI exits unexpectedly, start it again so watcher remains useful.
                print("\nUI process exited. Restarting...\n")
                child = _start_ui_process()
                previous_snapshot = _snapshot_mtimes(root)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_process(child)
        print("\nStopped.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
