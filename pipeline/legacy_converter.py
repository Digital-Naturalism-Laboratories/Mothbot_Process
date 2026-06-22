#!/usr/bin/env python3
"""
Legacy dataset converter for Mothbot Process.

Old Mothbot versions wrote outputs next to source images:
  <date_folder>/patches/<patch>.jpg
  <date_folder>/<image>_botdetection.json

Current versions mirror everything into _processed/:
  _processed/<rel>/<patch>.jpg   (flat, no patches/ subfolder)
  _processed/<rel>/<image>_botdetection.json

This module detects and converts legacy-layout collections.
"""

import glob
import json
import os
import shutil
from pathlib import Path

from core.paths import get_processed_folder


def is_legacy_collection(folder: str) -> bool:
    """Return True if *folder* looks like a legacy-format collection.

    Criteria: the folder contains .jpg source images AND either:
    - a ``patches/`` subdirectory exists with .jpg files, OR
    - ``*_botdetection.json`` files sit directly in the folder.
    """
    if not os.path.isdir(folder):
        return False

    has_source_jpgs = any(
        True for _ in glob.iglob(os.path.join(glob.escape(folder), "*.jpg"))
    ) or any(
        True for _ in glob.iglob(os.path.join(glob.escape(folder), "*.jpeg"))
    )
    if not has_source_jpgs:
        return False

    patches_dir = os.path.join(folder, "patches")
    has_patches_folder = os.path.isdir(patches_dir) and bool(
        glob.glob(os.path.join(glob.escape(patches_dir), "*.jpg"))
        or glob.glob(os.path.join(glob.escape(patches_dir), "*.jpeg"))
    )

    has_loose_jsons = bool(
        glob.glob(os.path.join(glob.escape(folder), "*_botdetection.json"))
    )

    return has_patches_folder or has_loose_jsons


def scan_dataset_for_legacy(dataset_root: str) -> list[dict]:
    """Walk *dataset_root* and return info about every legacy-format collection.

    Skips ``_processed/`` and hidden directories.  Returns a list of dicts:
    ``{source_folder, rel_path, json_count, patch_count,
       has_patches_dir, has_loose_jsons}``
    """
    results = []
    dataset_root = os.path.realpath(dataset_root)
    processed_marker = os.path.join(dataset_root, "_processed")

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        # Prune hidden dirs and the _processed mirror in-place so os.walk skips them.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and os.path.join(dirpath, d) != processed_marker
            and not os.path.join(dirpath, d).startswith(processed_marker + os.sep)
        ]

        if not is_legacy_collection(dirpath):
            continue

        patches_dir = os.path.join(dirpath, "patches")
        json_count = len(glob.glob(os.path.join(glob.escape(dirpath), "*_botdetection.json")))
        patch_count = (
            len(glob.glob(os.path.join(glob.escape(patches_dir), "*.jpg")))
            + len(glob.glob(os.path.join(glob.escape(patches_dir), "*.jpeg")))
        ) if os.path.isdir(patches_dir) else 0

        try:
            rel = os.path.relpath(dirpath, dataset_root)
        except ValueError:
            rel = dirpath

        results.append({
            "source_folder": dirpath,
            "rel_path": rel,
            "json_count": json_count,
            "patch_count": patch_count,
            "has_patches_dir": os.path.isdir(patches_dir),
            "has_loose_jsons": json_count > 0,
        })

    return results


def convert_collection(
    source_folder: str,
    dataset_root: str,
    *,
    delete_originals: bool = False,
) -> "Generator[str, None, None]":
    """Convert one legacy collection to the current _processed/ layout.

    Yields human-readable log lines as work progresses.
    """
    source_folder = os.path.realpath(source_folder)
    dataset_root = os.path.realpath(dataset_root)

    try:
        processed_folder = get_processed_folder(source_folder, dataset_root)
    except ValueError as exc:
        yield f"❌ Cannot compute output path: {exc}\n"
        return

    yield f"📁 Output folder: {processed_folder}\n"

    # ── Move / rewrite JSON files ─────────────────────────────────────────────
    json_paths = sorted(glob.glob(os.path.join(glob.escape(source_folder), "*_botdetection.json")))
    if json_paths:
        yield f"📄 Processing {len(json_paths)} JSON file(s)…\n"
    for json_path in json_paths:
        basename = os.path.basename(json_path)
        dest = os.path.join(processed_folder, basename)
        try:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            yield f"  ⚠️  Could not read {basename}: {exc}\n"
            continue

        # Update patch_path in every shape: strip leading "patches/" prefix.
        for shape in data.get("shapes", []):
            raw = shape.get("patch_path", "")
            if isinstance(raw, str) and raw.startswith("patches/"):
                shape["patch_path"] = raw[len("patches/"):]

        try:
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
            yield f"  ✓ {basename}\n"
        except Exception as exc:
            yield f"  ❌ Failed to write {dest}: {exc}\n"
            continue

        if delete_originals:
            try:
                os.remove(json_path)
            except Exception as exc:
                yield f"  ⚠️  Could not delete original {basename}: {exc}\n"

    # ── Move patch images ─────────────────────────────────────────────────────
    patches_dir = os.path.join(source_folder, "patches")
    if os.path.isdir(patches_dir):
        patch_files = sorted(
            glob.glob(os.path.join(glob.escape(patches_dir), "*.jpg"))
            + glob.glob(os.path.join(glob.escape(patches_dir), "*.jpeg"))
            + glob.glob(os.path.join(glob.escape(patches_dir), "*.png"))
        )
        yield f"🦋 Moving {len(patch_files)} patch image(s)…\n"
        for src in patch_files:
            fname = os.path.basename(src)
            dest = os.path.join(processed_folder, fname)
            try:
                shutil.copy2(src, dest)
                yield f"  ✓ {fname}\n"
            except Exception as exc:
                yield f"  ❌ Failed to copy {fname}: {exc}\n"

        if delete_originals and patch_files:
            try:
                shutil.rmtree(patches_dir)
                yield f"  🗑️  Removed patches/ folder\n"
            except Exception as exc:
                yield f"  ⚠️  Could not remove patches/ folder: {exc}\n"
    else:
        yield "  (no patches/ folder found — skipping patch images)\n"

    yield "✅ Done.\n"
