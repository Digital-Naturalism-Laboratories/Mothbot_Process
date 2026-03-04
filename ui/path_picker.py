"""Reusable native path picker helpers for UI modules."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Literal

logger = logging.getLogger("mothbot.ui.path_picker")

PickerMode = Literal["file", "folder"]


def browse_path(current_path: str = "", mode: PickerMode = "file", filetypes=None) -> str:
    chosen, _error_message = browse_path_with_status(
        current_path=current_path,
        mode=mode,
        filetypes=filetypes,
    )
    return chosen


def browse_path_with_status(
    current_path: str = "",
    mode: PickerMode = "file",
    filetypes=None,
) -> tuple[str, str]:
    """
    Open a native path picker dialog.

    Returns ``(path, error_message)``.

    ``path`` is empty when user cancels or no picker is usable.
    ``error_message`` is empty unless the picker backend fails.
    """
    initial_dir = _get_initial_dir(current_path)

    used_osascript = False
    if sys.platform == "darwin":
        used_osascript = True
        chosen, state = _browse_path_via_osascript(
            mode=mode, initial_dir=initial_dir, filetypes=filetypes,
        )
        if chosen:
            return chosen, ""
        if state == "cancelled":
            return "", ""
        logger.warning("AppleScript picker failed; falling back to Tk dialog")

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        if used_osascript:
            message = "No usable file picker: tkinter unavailable after AppleScript failure"
            logger.error(message)
        else:
            message = f"No usable file picker: tkinter unavailable on {sys.platform}"
            logger.error(message)
        return "", message

    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        if mode == "folder":
            chosen = filedialog.askdirectory(initialdir=initial_dir or None)
        else:
            chosen = filedialog.askopenfilename(
                initialdir=initial_dir or None,
                filetypes=filetypes or [("All files", "*.*")],
            )
        return chosen or "", ""
    except Exception:
        message = "Tk file picker failed. Check app permissions and try again."
        logger.exception("Tk file picker failed")
        return "", message
    finally:
        try:
            root.destroy()
        except Exception:
            pass


# -- Private helpers ---------------------------------------------------------


def _browse_path_via_osascript(
    mode: PickerMode,
    initial_dir: str = "",
    filetypes: list | None = None,
) -> tuple[str, str]:
    """Run a native macOS file/folder picker via AppleScript.

    Returns ``(path, state)`` where *state* is
    ``"selected"``, ``"cancelled"``, or ``"failed"``.
    """
    script = _build_osascript(mode, initial_dir, filetypes)
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").lower()
            if "user canceled" in stderr or "(-128)" in stderr:
                return "", "cancelled"
            logger.warning("osascript exited %d: %s", result.returncode, result.stderr.strip())
            return "", "failed"
        return (result.stdout or "").strip(), "selected"
    except Exception:
        logger.warning("osascript not available", exc_info=True)
        return "", "failed"


def _build_osascript(
    mode: PickerMode,
    initial_dir: str = "",
    filetypes: list | None = None,
) -> str:
    verb = "choose folder" if mode == "folder" else "choose file"
    clauses: list[str] = []

    if mode == "file" and filetypes:
        exts: list[str] = []
        for pattern in _iter_filetype_patterns(filetypes):
            for token in pattern.replace(";", " ").split():
                ext = token.lstrip("*.")
                if ext:
                    exts.append(ext)
        if exts:
            type_list = ", ".join(f'"{e}"' for e in exts)
            clauses.append(f"of type {{{type_list}}}")

    if initial_dir and os.path.isdir(initial_dir):
        escaped = initial_dir.replace("\\", "\\\\").replace('"', '\\"')
        clauses.append(f'default location (POSIX file "{escaped}")')

    prompt = "Select folder" if mode == "folder" else "Select file"
    clauses.append(f'with prompt "{prompt}"')

    inner = " ".join([verb] + clauses)
    return f"POSIX path of ({inner})"


def _iter_filetype_patterns(filetypes) -> list[str]:
    patterns: list[str] = []
    for item in filetypes:
        if len(item) < 2:
            continue
        maybe_patterns = item[1]
        if isinstance(maybe_patterns, str):
            patterns.append(maybe_patterns)
            continue
        if isinstance(maybe_patterns, (list, tuple)):
            patterns.extend(str(pattern) for pattern in maybe_patterns)
    return patterns


def _get_initial_dir(initial_path: str) -> str:
    if not initial_path:
        return ""
    if os.path.isdir(initial_path):
        return initial_path
    return os.path.dirname(initial_path)
