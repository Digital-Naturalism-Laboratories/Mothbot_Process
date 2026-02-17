"""Reusable native path picker helpers for UI modules."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Literal

PickerMode = Literal["file", "folder"]


def browse_path(current_path: str = "", mode: PickerMode = "file", filetypes=None) -> str:
    """
    Open a native path picker dialog.

    Returns an empty string when native dialogs are unavailable or user cancels.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        if sys.platform == "darwin":
            return _browse_path_via_osascript(mode=mode)
        return ""

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        initialdir = _get_initial_dir(current_path)
        if mode == "folder":
            chosen = filedialog.askdirectory(initialdir=initialdir or None)
        else:
            chosen = filedialog.askopenfilename(
                initialdir=initialdir or None,
                filetypes=filetypes or [("All files", "*.*")],
            )
    finally:
        root.destroy()

    return chosen or ""


def normalize_explorer_selection(selection) -> str:
    if isinstance(selection, list):
        return selection[0] if selection else ""
    return selection or ""


def _browse_path_via_osascript(mode: PickerMode) -> str:
    if mode == "folder":
        script = 'POSIX path of (choose folder with prompt "Select folder")'
    else:
        script = 'POSIX path of (choose file with prompt "Select file")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        return (result.stdout or "").strip()
    except Exception:
        return ""


def _get_initial_dir(initial_path: str) -> str:
    if not initial_path:
        return ""
    if os.path.isdir(initial_path):
        return initial_path
    return os.path.dirname(initial_path)
