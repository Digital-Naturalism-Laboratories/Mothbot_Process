# Desktop Packaging

This document covers CPU and CUDA packaging for the refactored in-process Mothbot UI.

## 1) Install Dependencies

Run all commands from the repository root (`mothbot-detect`).

Create and activate a packaging virtual environment first:

```bash
python3 -m venv .venv-packaging
source .venv-packaging/bin/activate
```

### CPU build profile (recommended default)

```bash
python3 -m pip install -e ".[cpu,packaging]"
```

### CUDA 11.8 build profile (optional)

```bash
python3 -m pip install -e ".[cuda118,packaging]" --extra-index-url https://download.pytorch.org/whl/cu118
```

Windows PowerShell:

```powershell
python -m venv .venv-packaging
.\.venv-packaging\Scripts\Activate.ps1
python -m pip install -e ".[cpu,packaging]"
```

You can also use the Make targets (same behavior, shorter commands):

```bash
make setup         # CPU dependencies
make gpu-setup     # optional CUDA 11.8 dependencies
```

## 2) Build Commands

### macOS

```bash
make build-macos
```

### Linux

```bash
make build-linux
```

### Windows (PowerShell)

```powershell
make build-windows
```

Direct script equivalents:

```bash
bash apps/scripts/build_desktop_macos.sh
bash apps/scripts/build_desktop_linux.sh
```

```powershell
.\apps\scripts\build_desktop_windows.ps1
```

You can also run PyInstaller directly (advanced):

```bash
python -m PyInstaller --clean --noconfirm apps/packaging/pyinstaller/mothbot_desktop.spec
```

## 3) Build Artifacts

Default output directory:

- macOS: `apps/dist/Mothbot.app`
- Linux/Windows: `apps/dist/Mothbot/`

Main executable:

- macOS: `apps/dist/Mothbot.app` (double-clickable app bundle)
- Linux: `apps/dist/Mothbot/Mothbot`
- Windows: `apps/dist/Mothbot/Mothbot.exe`

## 3.1) macOS distributable release files

After `make build-macos`, generate end-user artifacts:

```bash
make package-macos
```

This creates:

- `apps/release/Mothbot-<version>-macos-<arch>.zip`
- `apps/release/Mothbot-<version>-macos-<arch>.dmg`
- `apps/release/SHA256SUMS.txt`

## 3.2) Linux and Windows distributable release files

After `make build-linux` or `make build-windows`, the build scripts also package release artifacts:

- Linux: `apps/release/Mothbot-<version>-linux-<arch>.zip`
- Windows: `apps/release/Mothbot-<version>-windows-<arch>.zip`

Linux and Windows packaging use 7-Zip compression (`-tzip -mx=9`) to reduce artifact size while keeping `.zip` output.

## 4) ExifTool Requirement

`pipeline/insert_exif.py` requires `exiftool`.

Resolution order:

1. `MOTHBOT_EXIFTOOL_PATH` environment variable (if set)
2. Bundled paths inside frozen app (`_MEIPASS`, Windows bundle folder)
3. `exiftool` available in PATH

If Exif insertion fails, set:

```bash
export MOTHBOT_EXIFTOOL_PATH="/full/path/to/exiftool"
```

Windows PowerShell:

```powershell
$env:MOTHBOT_EXIFTOOL_PATH="C:\full\path\to\exiftool.exe"
```

## 5) Smoke Test Checklist

After build, verify:

1. App starts and Gradio UI loads.
2. Folder scan populates nightly folders.
3. `Detect` runs and writes `_botdetection.json`.
4. `ID` runs and writes classification metadata to JSON.
5. `Insert Metadata` applies CSV metadata fields.
6. `Cluster` runs (DINO model lazy-load works).
7. `Insert Exif` runs with exiftool available.
8. `Create Dataset` writes `samples.json`.
9. `Generate CSV` exports CSV from `samples.json`.

Useful cleanup:

```bash
make clean
```

## 6) Known Risk Areas

- Bundle size is large with Torch/FiftyOne.
- CUDA builds are platform-specific and should be released separately from CPU builds.
- FiftyOne/Mongo runtime behavior can vary across OS packaging environments.
- Cross-OS desktop packaging is not supported by our current toolchain; in practice, Windows artifacts must be built on a Windows runner (we could not produce a reliable Windows build from macOS).

## 7) Desktop startup logs

The desktop entrypoint writes startup logs to disk, including uncaught exceptions that can happen before the UI opens.

Default log locations:

- macOS: `~/Library/Logs/Mothbot/desktop.log`
- Linux: `~/.local/state/mothbot/desktop.log`
- Windows: `%LOCALAPPDATA%\Mothbot\logs\desktop.log`

If the app icon bounces and the app closes, open this file and check the most recent stack trace at the end.

## 8) Release a new version (GitHub tags + Actions)

Desktop release builds for all three platforms (macOS, Linux, Windows) are run when a Git tag is pushed:

- Workflow: `.github/workflows/windows-desktop-build.yml`
- Trigger: `push` tags matching `v*` (for example `v0.8.0`)
- Output: a GitHub Release with uploaded build artifacts and a unified `SHA256SUMS.txt`

The workflow can also be started manually with `workflow_dispatch`; manual runs upload build artifacts but do not publish a GitHub Release.

Recommended release checklist:

1. Ensure `pyproject.toml` has the new version.
2. Commit version-related changes and merge to `main`.
3. Create and push an annotated tag that matches the version:

```bash
git checkout main
git pull
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

4. Wait for the GitHub Actions workflow to finish for all 3 runners.
5. Open GitHub Releases and verify the new release contains:
   - macOS artifacts (`.zip`, `.dmg`)
   - Linux artifact (`.zip`)
   - Windows artifact (`.zip`)
   - `SHA256SUMS.txt`
6. Optionally validate one checksum locally after download:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Size policy:

- Warning at 1.6 GiB per artifact
- Hard fail at 1.8 GiB per artifact

Notes:

- Keep the tag version and `pyproject.toml` version aligned.
- The size budget leaves headroom under GitHub Releases' 2 GiB per-asset API limit.
