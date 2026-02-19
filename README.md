# mothbot-detect

`mothbot-detect` is a Gradio-based desktop workflow for moth image processing.

Main pipeline steps:

- detection (`pipeline/detect.py`)
- identification (`pipeline/identify.py`)
- metadata insertion (`pipeline/insert_metadata.py`)
- clustering (`pipeline/cluster.py`)
- exif insertion (`pipeline/insert_exif.py`)
- dataset creation (`pipeline/create_dataset.py`)
- CSV export (`pipeline/export_csv.py`)

## Quick Start (Local Development)

Requirements:

- Python 3.11+
- `make`

From the repository root:

```bash
make setup
make run
```

With auto-reload during development:

```bash
make run-watch
```

Optional CUDA 11.8 setup:

```bash
make gpu-setup
```

## Project Structure

- `ui/`: Gradio app and local runners
- `pipeline/`: detection and post-processing steps
- `core/`: shared utilities
- `apps/`: desktop entrypoints, packaging specs, and build scripts

Packaged desktop entrypoint:

- `apps/desktop_main.py`

## Common Commands

- `make help`: top-level command guide
- `make dev-help`: development targets
- `make release-help`: packaging/release targets
- `make clean`: remove packaging artifacts and caches

## Documentation

- Desktop packaging and release: `docs/DESKTOP_PACKAGING.md`

## Desktop Releases (Tag-Based CI)

Desktop release artifacts are built in GitHub Actions when you push a version tag (for example `v0.8.0`).

Release flow:

1. Merge release changes to `main`.
2. Create and push a version tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"` then `git push origin vX.Y.Z`.
3. Wait for the GitHub workflow to publish the release artifacts.

For full release and local packaging steps, see `docs/DESKTOP_PACKAGING.md`.
