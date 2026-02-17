# mothbot-detect

`mothbot-detect` is a Gradio-based desktop workflow for:

- detection (`pipeline/detect.py`)
- identification (`pipeline/identify.py`)
- metadata insertion (`pipeline/insert_metadata.py`)
- clustering (`pipeline/cluster.py`)
- exif insertion (`pipeline/insert_exif.py`)
- dataset creation (`pipeline/create_dataset.py`)
- CSV export (`pipeline/export_csv.py`)

Lightweight module namespaces are available for cleaner structure:

- UI: `ui/`
- pipelines: `pipeline/`
- shared utilities: `core/`

Local (non-packaged) Gradio run:

- `python -m ui.local_main`
- `make setup` (first time)
- `make run`

Packaged desktop entrypoint:

- `apps/desktop_main.py`

Makefile usage:

- `make help` (router)
- `make dev-help` / `make release-help`

See `docs/DESKTOP_PACKAGING.md` for packaging and build instructions.
For GitHub organization publishing, see `docs/PUBLISH_GITHUB.md`.
