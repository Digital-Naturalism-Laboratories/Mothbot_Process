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
