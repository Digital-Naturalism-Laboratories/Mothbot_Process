# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Mothbot desktop builds.

Usage:
  python -m PyInstaller --clean --noconfirm apps/packaging/pyinstaller/mothbot_desktop.spec
"""

from pathlib import Path
import importlib.util
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_dir = Path(SPECPATH).resolve().parents[2]
ai_dir = project_dir.parent

hiddenimports = []
for package in [
    "gradio",
    "gradio_client",
    "ultralytics",
    "fiftyone",
    "fiftyone_brain",
    "open_clip",
    "bioclip",
]:
    hiddenimports += collect_submodules(package)

datas = []
for package in [
    "gradio",
    "gradio_client",
    "groovy",
    "ultralytics",
    "fiftyone",
    "fiftyone_brain",
    "safehttpx",
    "open_clip",
    "bioclip",
]:
    datas += collect_data_files(package)


def _package_dir(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec.origin:
        return Path(spec.origin).parent
    return None


# Some Gradio internals read their own source files at runtime.
# Include full package trees so those files are available in frozen mode.
for package_name in ["gradio", "safehttpx", "groovy", "open_clip", "bioclip"]:
    pkg_dir = _package_dir(package_name)
    if pkg_dir and pkg_dir.exists():
        datas.append((str(pkg_dir), package_name))

# Optional local files/directories that may or may not exist in every checkout.
for src in [
    project_dir / "artifacts",
    project_dir / "assets" / "favicon.png",
    project_dir / "Mothbox_Main_Metadata_Field_Sheet_Example - Form responses 1.csv",
    project_dir / "SpeciesList_CountryIndonesia_TaxaInsecta_doi.org10.15468dl.8p8wua.csv",
    ai_dir / "exiftool-13.32_64",
]:
    if src.exists():
        if src.is_dir():
            datas.append((str(src), src.name))
        else:
            datas.append((str(src), "."))

a = Analysis(
    [str(project_dir / "apps" / "desktop_main.py")],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Mothbot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Mothbot",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Mothbot.app",
        bundle_identifier="org.mothbox.desktop",
        info_plist={
            "CFBundleName": "Mothbot",
            "CFBundleDisplayName": "Mothbot",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
