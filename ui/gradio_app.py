#!/usr/bin/env python3
"""
Mothbot Gradio UI – desktop-packaging-friendly version.

Key changes from the subprocess-based original:
  * Worker scripts are called via their ``run()`` functions (in-process).
  * stdout is captured via ``core.common.run_in_thread`` and streamed into
    Gradio Textbox outputs — same UX, no subprocess overhead.
  * Path fields support both paste/type and optional native browse dialogs.
"""

import os
import re
import glob
import sys
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import tomllib
import gradio as gr

from core.common import run_in_thread, request_cancel
from ui.tray import start_tray
from ui.single_instance import ensure_single_instance
from ui.path_picker import browse_path, browse_path_with_status

# Lazy-import worker modules so heavy ML deps only load when a tab is used.
from pipeline import cluster as Mothbot_Cluster
from pipeline import detect as Mothbot_Detect
from pipeline import identify as Mothbot_ID
from pipeline import insert_exif as Mothbot_InsertExif
from pipeline import insert_metadata as Mothbot_InsertMetadata

TAXA_COLS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(
    os.getenv("MOTHBOT_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
)


def _normalize_version(raw_version):
    normalized = (raw_version or "").strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized or "dev"


def _read_version_file(version_file):
    try:
        if version_file.exists():
            return _normalize_version(version_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _get_git_tag_version():
    if not (PROJECT_ROOT / ".git").exists():
        return None

    git_commands = [
        ["git", "describe", "--tags", "--exact-match"],
        ["git", "describe", "--tags", "--abbrev=0"],
    ]
    for command in git_commands:
        try:
            raw_output = subprocess.check_output(
                command,
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return _normalize_version(raw_output)
        except Exception:
            continue
    return None


def _get_app_version():
    env_version = _normalize_version(os.getenv("MOTHBOT_RELEASE_VERSION", ""))
    if env_version != "dev":
        return env_version

    # Prefer an explicitly bundled version file so packaged apps can show
    # the exact GitHub release version used during the build.
    version_candidates = [Path(sys.executable).resolve().parent / "VERSION", PROJECT_ROOT / "VERSION"]
    maybe_meipass = getattr(sys, "_MEIPASS", None)
    if maybe_meipass:
        version_candidates.insert(0, Path(maybe_meipass) / "VERSION")

    for candidate in version_candidates:
        file_version = _read_version_file(candidate)
        if file_version:
            return file_version

    git_tag_version = _get_git_tag_version()
    if git_tag_version:
        return git_tag_version

    try:
        return _normalize_version(version("mothbot"))
    except Exception:
        pass

    try:
        pyproject_data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        return _normalize_version(pyproject_data["project"]["version"])
    except Exception:
        return "dev"


def _get_platform_label():
    system_name = platform.system().lower()
    if system_name == "darwin":
        return "macOS"
    if system_name == "windows":
        return "Windows"
    if system_name == "linux":
        return "Linux"
    return platform.system() or "Unknown OS"


APP_META_LABEL = f"v{_get_app_version()} | {_get_platform_label()}"


# ──────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────


def app():
    with gr.Blocks(
        title="Mothbot",
        css="""
            /* Setup - neutral white */
            button.svelte-1tcem6n:nth-child(1).selected {
                background-color: #e0e0e0 !important;
                color: #000000 !important;
            }
            /* Process - orange (run all) */
            button.svelte-1tcem6n:nth-child(2).selected {
                background-color: #ff8c00 !important;
                color: #ffffff !important;
            }
            /* Detect - red (step 1) */
            button.svelte-1tcem6n:nth-child(3).selected {
                background-color: #ff4444 !important;
                color: #ffffff !important;
            }
            /* Cluster - blue (step 2) */
            button.svelte-1tcem6n:nth-child(4).selected {
                background-color: #4488ff !important;
                color: #ffffff !important;
            }
            /* ID - green (step 3 group) */
            button.svelte-1tcem6n:nth-child(5).selected {
                background-color: #22ff88 !important;
                color: #ffffff !important;
            }
            /* Insert Metadata - green (step 3 group) */
            button.svelte-1tcem6n:nth-child(6).selected {
                background-color: #22ff88 !important;
                color: #ffffff !important;
            }
            /* Insert Exif - green (step 3 group) */
            button.svelte-1tcem6n:nth-child(7).selected {
                background-color: #22ff88 !important;
                color: #ffffff !important;
            }
            /* Unselected tab color hints */
            button.svelte-1tcem6n:nth-child(3):not(.selected) {
                border-bottom: 3px solid #ff4444 !important;
            }
            button.svelte-1tcem6n:nth-child(4):not(.selected) {
                border-bottom: 3px solid #4488ff !important;
            }
            button.svelte-1tcem6n:nth-child(5):not(.selected),
            button.svelte-1tcem6n:nth-child(6):not(.selected),
            button.svelte-1tcem6n:nth-child(7):not(.selected) {
                border-bottom: 3px solid #44ff44 !important;
            }
            #app-meta-row {
                justify-content: flex-end;
                margin-top: 10px;
            }
            #app-meta-badge p {
                margin: 0;
                font-size: 12px;
                color: #666666;
                background: #f4f4f4;
                border: 1px solid #d8d8d8;
                border-radius: 999px;
                padding: 4px 10px;
                line-height: 1.2;
            }
        """,
    ) as demo:
        mapping_state = gr.State({})
        dataset_root_state = gr.State("")  # top-level chosen folder
        toggle_label_state = gr.State("Select All")
        picker_error_state = gr.State("")
        selected_paths = gr.JSON(
            label="Confirmed Image Collections to be Processed", visible=False
        )
        # Tracks which selected keys are externally-processed (no source images)
        external_keys_state = gr.State(set())

        # ── Global Action Bar (Stop + Quit) – declared before tabs to maintain scope and render order ──
        with gr.Row():
            stop_btn = gr.Button(
                "Stop Current Run", variant="stop", size="sm", scale=0, min_width=200,
                visible=False,
            )
            gr.HTML("<div style='flex:1'></div>")  # spacer
            quit_btn = gr.Button("Quit Mothbot", variant="stop", size="sm", scale=0, min_width=160)
            quit_confirm_row = gr.Row(visible=False)
            with quit_confirm_row:
                quit_yes_btn = gr.Button("Yes, quit", variant="stop",     size="sm", scale=0, min_width=120)
                quit_no_btn  = gr.Button("Cancel",    variant="secondary", size="sm", scale=0, min_width=100)

        with gr.Tabs(selected="setup") as main_tabs:
            # ~~~~~~~~~~~~ Setup TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Setup", id="setup"):
                advanced_mode = gr.Checkbox(
                        label="Advanced mode",
                        value=False,
                        scale=0,
                        min_width=150,
                        container=False,
                    )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "### Datasets Folder: Pick a folder of your datasets to process"
                        )
                        deployment_path = gr.Text(
                            label="Datasets Folder Path (paste or type)",
                            placeholder="/path/to/your/deployment/folder",
                            interactive=True,
                        )
                        deployment_browse_btn = gr.Button(
                            "Pick a Datasets Folder", size="sm", variant="primary"
                        )
                        with gr.Group():
                            status = gr.Textbox(
                                label="Error", lines=3, interactive=False, visible=False
                            )
                            folder_choices = gr.CheckboxGroup(
                                label="Image Collections Found (select which to process)",
                                choices=[],
                                value=[],
                                interactive=True,
                                visible=False,
                            )
                            toggle_all_btn = gr.Button(
                                "Select All", size="sm", visible=False
                            )
                        continue_process_btn = gr.Button(
                            "Continue to Process",
                            variant="primary",
                            interactive=False,
                            visible=False,
                        )

                    with gr.Column():
                        gr.Markdown("### Additional Processing Files:")
                        with gr.Row():
                            yolo_model_path = gr.Text(
                                value=DEFAULT_YOLO_MODEL,
                                label="Detection Model Path",
                            )
                            yolo_browse_btn = gr.Button("Browse", size="sm", scale=0, min_width=100)
                        with gr.Row():
                            with gr.Column():
                                species_path = gr.Text(
                                    label="Species List:",
                                    value=DEFAULT_SPECIES_CSV,
                                )
                                species_browse_btn = gr.Button("Browse", size="sm")
                            with gr.Column():
                                metadata_csv_file = gr.Text(
                                    label="metadata field sheet:",
                                    value=DEFAULT_METADATA_CSV,
                                )
                                metadata_browse_btn = gr.Button("Browse", size="sm")


                deployment_browse_btn.click(
                    fn=browse_deployment_folder,
                    inputs=[deployment_path],
                    outputs=[deployment_path, picker_error_state],
                ).then(
                    fn=scan_deployment_folder,
                    inputs=[deployment_path, picker_error_state],
                    outputs=[
                        status,
                        folder_choices,
                        mapping_state,
                        toggle_label_state,
                        continue_process_btn,
                        selected_paths,
                        toggle_all_btn,
                        external_keys_state,
                        dataset_root_state,
                    ],
                )
                deployment_path.change(
                    fn=scan_deployment_folder_on_change,
                    inputs=[deployment_path],
                    outputs=[
                        status,
                        folder_choices,
                        mapping_state,
                        toggle_label_state,
                        continue_process_btn,
                        selected_paths,
                        toggle_all_btn,
                        external_keys_state,
                        dataset_root_state,
                    ],
                )
                metadata_browse_btn.click(
                    fn=browse_metadata_csv,
                    inputs=[metadata_csv_file],
                    outputs=[metadata_csv_file],
                )
                species_browse_btn.click(
                    fn=browse_species_csv,
                    inputs=[species_path],
                    outputs=[species_path],
                )
                yolo_browse_btn.click(
                    fn=browse_yolo_model,
                    inputs=[yolo_model_path],
                    outputs=[yolo_model_path],
                )

                toggle_all_btn.click(
                    fn=toggle_select_all,
                    inputs=[folder_choices, mapping_state, toggle_label_state],
                    outputs=[folder_choices, toggle_label_state],
                ).then(
                    fn=confirm_selection,
                    inputs=[folder_choices, mapping_state, external_keys_state, dataset_root_state],
                    outputs=[selected_paths, continue_process_btn],
                )
                toggle_label_state.change(
                    lambda lbl: gr.update(value=lbl),
                    inputs=toggle_label_state,
                    outputs=toggle_all_btn,
                )
                folder_choices.change(
                    fn=confirm_selection,
                    inputs=[folder_choices, mapping_state, external_keys_state, dataset_root_state],
                    outputs=[selected_paths, continue_process_btn],
                )
            # ~~~~~~~~~~~~ PROCESS TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Process", id="process") as process_tab:
                process_output_box = gr.Textbox(
                    label="Process Output", lines=20, interactive=False
                )

            # ~~~~~~~~~~~~ DETECTION TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Detect", id="detect", visible=False) as detect_tab:
                with gr.Row():
                    det_model_path_mirror = gr.Text(
                        label="Detection Model Path",
                        interactive=True,
                    )
                    det_model_browse_mirror = gr.Button("Browse", size="sm", scale=0, min_width=100)
                with gr.Row():
                    imgsz = gr.Number(
                        label="Yolo processing img size (should be same as yolo model) (leave default)",
                        value=1600,
                    )
                    OVERWRITE_PREV_BOT_DETECTIONS = gr.Checkbox(
                        value=True,
                        label="Overwrite any previous Bot Detections (Create new detection files)",
                    )
                DET_run_btn = gr.Button("Run Detection", variant="primary")
                DET_output_box = gr.Textbox(label="Detection Output", lines=20)

                continue_cluster_btn = gr.Button(
                    "Continue to Cluster", variant="primary", interactive=False
                )

                DET_run_btn.click(
                    fn=run_detection_with_continue,
                    inputs=[
                        selected_paths,
                        yolo_model_path,
                        imgsz,
                        OVERWRITE_PREV_BOT_DETECTIONS,
                        external_keys_state,
                    ],
                    outputs=[DET_output_box, continue_cluster_btn, stop_btn],
                )

                continue_cluster_btn.click(
                    fn=go_to_cluster_tab,
                    inputs=[],
                    outputs=[main_tabs],
                )

            # ~~~~~~~~~~~~ Cluster Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Cluster Perceptually", id="cluster", visible=False) as cluster_tab:
                cluster_run_btn = gr.Button("Cluster Perceptually", variant="primary")
                cluster_output_box = gr.Textbox(label="Cluster Output", lines=20)
                continue_id_btn = gr.Button(
                    "Continue to ID", variant="primary", interactive=False
                )
                cluster_run_btn.click(
                    fn=run_cluster_with_continue,
                    inputs=[selected_paths],
                    outputs=[cluster_output_box, continue_id_btn, stop_btn],
                )

                continue_id_btn.click(
                    fn=go_to_id_tab,
                    inputs=[],
                    outputs=[main_tabs],
                )

            # ~~~~~~~~~~~~ IDENTIFICATION TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("ID", id="id", visible=False) as id_tab:
                with gr.Row():
                    with gr.Column():
                        radio = gr.Radio(
                            TAXA_COLS,
                            label="Select how deep you want to try to automatically Identify:",
                            type="value",
                            value="order",
                        )
                        with gr.Column():
                            taxa_output = gr.Number(
                                label="Taxa Index",
                                value=TAXA_COLS.index("order"),
                                visible=False,
                            )
                            radio.change(get_index, inputs=radio, outputs=taxa_output)

                    with gr.Column():
                        ID_HUMANDETECTIONS = gr.Checkbox(
                            value=True,
                            label="Identify Human Detections (Leave as True)",
                        )
                        ID_BOTDETECTIONS = gr.Checkbox(
                            value=True, label="Identify Bot Detections (Leave as True)"
                        )
                        OVERWRITE_PREV_BOT_IDENTIFICATIONS = gr.Checkbox(
                            value=True,
                            label="OVERWRITE_PREVIOUS_BOT_IDENTIFICATIONS (Create new automated IDs)",
                        )

                with gr.Row():
                    id_species_mirror = gr.Text(
                        label="Species List:",
                        interactive=True,
                    )
                    id_species_browse_mirror = gr.Button("Browse", size="sm", scale=0, min_width=100)
                ID_run_btn = gr.Button("Run Identification", variant="primary")
                ID_output_box = gr.Textbox(label="Identification Output", lines=20)

                ID_run_btn.click(
                    fn=run_ID,
                    inputs=[
                        selected_paths,
                        id_species_mirror,
                        taxa_output,
                        ID_HUMANDETECTIONS,
                        ID_BOTDETECTIONS,
                        OVERWRITE_PREV_BOT_IDENTIFICATIONS,
                    ],
                    outputs=[ID_output_box, stop_btn],
                )

            # ~~~~~~~~~~~~ Metadata Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Insert Metadata", id="metadata", visible=False) as metadata_tab:
                with gr.Row():
                    meta_csv_mirror = gr.Text(
                        label="Metadata field sheet:",
                        interactive=True,
                    )
                    meta_browse_mirror = gr.Button("Browse", size="sm", scale=0, min_width=100)
                metadata_run_btn = gr.Button("Insert Metadata", variant="primary")
                metadata_output_box = gr.Textbox(
                    label="Insert Metadata Output", lines=20
                )

                metadata_run_btn.click(
                    fn=run_metadata,
                    inputs=[selected_paths, meta_csv_mirror],
                    outputs=[metadata_output_box, stop_btn],
                )

            # ~~~~~~~~~~~~ Exif Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Insert Exif", id="exif", visible=False) as exif_tab:
                exif_run_btn = gr.Button("Insert Exif (Optional)", variant="primary")
                exif_output_box = gr.Textbox(label="Insert Exif Output", lines=20)

                exif_run_btn.click(
                    fn=run_exif,
                    inputs=[selected_paths],
                    outputs=[exif_output_box, stop_btn],
                )
            advanced_mode.change(
                fn=toggle_advanced_mode,
                inputs=[advanced_mode],
                outputs=[
                    detect_tab,
                    id_tab,
                    metadata_tab,
                    cluster_tab,
                    exif_tab,
                    process_tab,
                    main_tabs,
                ],
            )
            continue_process_btn.click(
                fn=go_to_process_tab,
                inputs=[],
                outputs=[main_tabs],
            ).then(
                fn=run_full_process,
                inputs=[
                    selected_paths,
                    yolo_model_path,
                    imgsz,
                    OVERWRITE_PREV_BOT_DETECTIONS,
                    species_path,
                    taxa_output,
                    ID_HUMANDETECTIONS,
                    ID_BOTDETECTIONS,
                    OVERWRITE_PREV_BOT_IDENTIFICATIONS,
                    metadata_csv_file,
                ],
                outputs=[process_output_box, stop_btn],
            )

        # ── Stop button ────────────────────────────────────────────────────────
        def do_cancel():
            request_cancel()
            return gr.update(value="⛔ Stopping…", interactive=False)

        stop_btn.click(fn=do_cancel, inputs=[], outputs=[stop_btn])

        # ── Cross-tab two-way sync (wired after all tabs so all components exist) ──
        # Setup ↔ Detect: Detection Model Path
        yolo_model_path.change(
            lambda v: v, inputs=[yolo_model_path], outputs=[det_model_path_mirror]
        )
        det_model_path_mirror.change(
            lambda v: v, inputs=[det_model_path_mirror], outputs=[yolo_model_path]
        )
        det_model_browse_mirror.click(
            fn=browse_yolo_model,
            inputs=[det_model_path_mirror],
            outputs=[det_model_path_mirror],
        ).then(
            lambda v: v, inputs=[det_model_path_mirror], outputs=[yolo_model_path]
        )

        # Setup ↔ ID: Species List
        species_path.change(
            lambda v: v, inputs=[species_path], outputs=[id_species_mirror]
        )
        id_species_mirror.change(
            lambda v: v, inputs=[id_species_mirror], outputs=[species_path]
        )
        id_species_browse_mirror.click(
            fn=browse_species_csv,
            inputs=[id_species_mirror],
            outputs=[id_species_mirror],
        ).then(
            lambda v: v, inputs=[id_species_mirror], outputs=[species_path]
        )

        # Setup ↔ Metadata: Metadata CSV
        metadata_csv_file.change(
            lambda v: v, inputs=[metadata_csv_file], outputs=[meta_csv_mirror]
        )
        meta_csv_mirror.change(
            lambda v: v, inputs=[meta_csv_mirror], outputs=[metadata_csv_file]
        )
        meta_browse_mirror.click(
            fn=browse_metadata_csv,
            inputs=[meta_csv_mirror],
            outputs=[meta_csv_mirror],
        ).then(
            lambda v: v, inputs=[meta_csv_mirror], outputs=[metadata_csv_file]
        )

        def ask_confirm():
            return gr.update(visible=False), gr.update(visible=True)

        def cancel_quit():
            return gr.update(visible=True), gr.update(visible=False)

        def quit_app():
            import signal
            import threading
            threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            return gr.update(value="Mothbot is shutting down — you can now close this browser tab.", interactive=False), gr.update(visible=False)

        quit_btn.click(fn=ask_confirm,    inputs=[], outputs=[quit_btn, quit_confirm_row])
        quit_no_btn.click(fn=cancel_quit, inputs=[], outputs=[quit_btn, quit_confirm_row])
        quit_yes_btn.click(fn=quit_app,   inputs=[], outputs=[quit_yes_btn, quit_confirm_row])
        
        with gr.Row(elem_id="app-meta-row"):
            gr.Markdown(APP_META_LABEL, elem_id="app-meta-badge")

    return demo


# ──────────────────────────────────────────────────────────────
#  Functions called by the UI
# ──────────────────────────────────────────────────────────────


def browse_deployment_folder(current_path):
    selected_path, picker_error = browse_path_with_status(
        current_path=current_path,
        mode="folder",
    )
    return (selected_path or current_path), picker_error


def browse_metadata_csv(current_path):
    return _browse_file(
        current_path, filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )


def browse_species_csv(current_path):
    return _browse_file(
        current_path, filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )


def browse_yolo_model(current_path):
    return _browse_file(
        current_path, filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
    )


def scan_deployment_folder(folder_path, picker_error_message=""):
    """Scan *folder_path* for image collections (raw and externally-processed)
    and return UI updates.

    Rules:
    - Raw collections: folders under *folder_path* (outside _processed/) that
      contain .jpg files.
    - Externally-processed collections: folders under *folder_path*/_processed/
      that contain .jpg patch images but whose corresponding raw folder has NO
      source images (so the collaborator only provided patches, not originals).
    - If a raw folder has source images, its _processed mirror is NOT shown as
      a separate entry (the raw folder already covers it).
    """
    _EMPTY = (
        gr.update(visible=True),
        gr.update(choices=[], value=[], visible=False),
        {},
        "Select All",
        gr.update(interactive=False, visible=False),
        [],
        gr.update(visible=False),
        set(),
        "",  # dataset_root_state
    )

    if picker_error_message:
        return (gr.update(value=f"Picker error: {picker_error_message}", visible=True),) + _EMPTY[1:]

    if not folder_path or not os.path.isdir(folder_path):
        return (gr.update(value="No valid folder path provided.", visible=True),) + _EMPTY[1:]

    # ── Raw collections ──────────────────────────────────────────────────────
    raw_matches = find_image_collections(folder_path)

    # Build a set of relative paths that have raw source images, so we can
    # suppress their _processed mirror from appearing as an external entry.
    raw_rel_paths = set()
    for p in raw_matches:
        try:
            raw_rel_paths.add(os.path.relpath(p, folder_path))
        except ValueError:
            pass

    # ── Externally-processed collections ─────────────────────────────────────
    processed_root = os.path.join(folder_path, "_processed")
    external_matches = []
    if os.path.isdir(processed_root):
        ext_collections = find_image_collections(processed_root)
        for p in ext_collections:
            try:
                rel_under_processed = os.path.relpath(p, processed_root)
            except ValueError:
                continue
            # Only include if the corresponding raw folder has no source images
            if rel_under_processed not in raw_rel_paths:
                external_matches.append(p)

    if not raw_matches and not external_matches:
        return (
            gr.update(value=f"No folders containing images found in:\n{folder_path}", visible=True),
        ) + _EMPTY[1:]

    choices = []
    mapping = {}
    external_keys = set()
    seen_values = set()

    def _make_entry(p, folder_root, label_prefix, is_external):
        try:
            rel = os.path.relpath(p, folder_path)
        except ValueError:
            rel = p
        value = rel if rel != "." else os.path.basename(p)
        i = 1
        orig = value
        while value in seen_values:
            value = f"{orig} ({i})"
            i += 1
        seen_values.add(value)

        jpeg_count = _count_matching_files(p, ("*.jpg", "*.jpeg"))

        if is_external:
            # For external collections the folder IS the processed mirror —
            # count patches (all jpgs) and JSONs directly inside it.
            json_count  = _count_matching_files(p, ("*.json",))
            label = f"⚡ {label_prefix}  ({jpeg_count} patches"
            if json_count:
                label += f", {json_count} detections"
            label += ")  [externally processed — no source images]"
        else:
            processed_mirror = os.path.join(folder_path, "_processed", os.path.relpath(p, folder_path))
            json_count   = _count_matching_files(processed_mirror, ("*.json",))   if os.path.isdir(processed_mirror) else 0
            patch_count  = _count_matching_files(processed_mirror, ("*.jpg", "*.jpeg")) if os.path.isdir(processed_mirror) else 0
            patch_count  = max(0, patch_count)
            label = f"{label_prefix}  ({jpeg_count} images"
            if json_count:
                label += f", {json_count} detections"
            if patch_count:
                label += f", {patch_count} patches"
            label += ")"

        choices.append((label, value))
        mapping[value] = os.path.abspath(p)
        if is_external:
            external_keys.add(value)

    for p in raw_matches:
        try:
            rel = os.path.relpath(p, folder_path)
        except ValueError:
            rel = str(p)
        display = rel if rel != "." else os.path.basename(p)
        _make_entry(p, folder_path, display, is_external=False)

    for p in external_matches:
        try:
            rel_from_processed = os.path.relpath(p, processed_root)
        except ValueError:
            rel_from_processed = str(p)
        display = f"_processed/{rel_from_processed}"
        _make_entry(p, processed_root, display, is_external=True)

    status = (
        f"Selected folder: {folder_path}\n"
        f"Found {len(raw_matches)} raw collection(s)"
        + (f" + {len(external_matches)} externally-processed collection(s)." if external_matches else ".")
    )
    return (
        gr.update(value="", visible=False),
        gr.update(choices=choices, value=[], visible=True),
        mapping,
        "Select All",
        gr.update(interactive=False, visible=True),
        [],
        gr.update(visible=True),
        external_keys,
        folder_path,  # dataset_root_state
    )


def scan_deployment_folder_on_change(folder_path):
    return scan_deployment_folder(folder_path, "")


def toggle_select_all(current_values, mapping, button_label):
    del current_values
    if button_label == "Select All":
        return gr.update(value=list(mapping.keys())), "Deselect All"
    return gr.update(value=[]), "Select All"


def confirm_selection(selected_labels, mapping, external_keys=None, dataset_root=""):
    """Resolve selected checkbox labels to absolute folder paths.

    Returns a list of dicts: {"path": str, "external": bool}
    so downstream runners know which collections lack source images.
    """
    if not selected_labels:
        return [], gr.update(interactive=False)
    external_keys = external_keys or set()
    resolved = [
        {
            "path": mapping[label],
            "external": label in external_keys,
            "dataset_root": dataset_root or mapping[label],
        }
        for label in selected_labels
        if label in mapping
    ]
    return resolved, gr.update(interactive=bool(resolved))


def go_to_process_tab():
    return gr.Tabs(selected="process")


def go_to_id_tab():
    return gr.Tabs(selected="id")

def go_to_cluster_tab():
    return gr.Tabs(selected="cluster")

def toggle_advanced_mode(enabled):
    visible = bool(enabled)
    selected_tab = "setup" if visible else "process"
    return (
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=not visible),
        gr.Tabs(selected=selected_tab),
    )


def get_index(selected_word):
    return TAXA_COLS.index(selected_word)


def run_detection_with_continue(selected_folders, yolo_model, imsz, overwrite_bot, external_keys=None):
    SHOW_STOP = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP = gr.update(visible=False)
    if not selected_folders:
        yield "No image collections selected.\n", gr.update(interactive=False)
        return

    external_keys = external_keys or set()
    output_log = ""
    had_error = False

    for entry in selected_folders:
        folder       = entry["path"]                     if isinstance(entry, dict) else entry
        is_ext       = entry.get("external", False)       if isinstance(entry, dict) else False
        dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

        if is_ext:
            output_log += f"⚠️  Skipping detection for externally-processed collection (no source images):\n    {folder}\n"
            yield output_log, gr.update(interactive=False), SHOW_STOP
            continue

        output_log += f"---🕵🏾‍♀️ Running detection for {folder} ---\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP

        try:
            for chunk in run_in_thread(
                Mothbot_Detect.run,
                input_path=folder,
                yolo_model=yolo_model,
                imgsz=int(imsz),
                overwrite_prev_bot_detections=bool(overwrite_bot),
                dataset_root=dataset_root,
            ):
                output_log += chunk
                yield output_log, gr.update(interactive=False), SHOW_STOP
            output_log += f"✅ Detection completed for {folder}\n"
        except Exception as exc:
            had_error = True
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP

    output_log += "----------- Finished running Batch --------------"
    yield output_log, gr.update(interactive=(not had_error)), HIDE_STOP


def run_ID(selected_folders, species_list, chosenrank, IDHum, IDBot, overwrite_bot):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_ID.run,
        start_message="---🔍 Running IDENTIFICATION for {folder} ---\n",
        success_message="✅ Identification completed for {folder}\n",
        finish_message="------ ID processing finished ------",
        kwargs_builder=lambda folder, dataset_root: {
            "input_path": folder,
            "taxa_csv": species_list,
            "rank": int(chosenrank),
            "ID_Hum": bool(IDHum),
            "ID_Bot": bool(IDBot),
            "overwrite_prev_bot_ID": bool(overwrite_bot),
            "dataset_root": dataset_root,
        },
    )


def run_metadata(selected_folders, metadata):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_InsertMetadata.run,
        start_message="---🔍 Running METADATA for {folder} ---\n",
        success_message="✅ Insert Metadata completed for {folder}\n",
        finish_message="------ Insert Metadata processing finished ------",
        kwargs_builder=lambda folder, dataset_root: {
            "input_path": folder,
            "metadata_path": str(metadata),
            "dataset_root": dataset_root,
        },
    )


def run_cluster_with_continue(selected_folders):
    SHOW_STOP = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP = gr.update(visible=False)
    if not selected_folders:
        yield "No image collections selected.\n", gr.update(interactive=False), gr.update(visible=False)
        return

    output_log = ""
    had_error = False

    for entry in selected_folders:
        folder       = entry["path"]                     if isinstance(entry, dict) else entry
        is_ext       = entry.get("external", False)       if isinstance(entry, dict) else False
        dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

        output_log += f"---🔍 Running Cluster for {folder} ---\n"
        if is_ext:
            output_log += "  ℹ️  Externally-processed collection detected — building stub JSONs from patches before clustering...\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP

        try:
            if is_ext:
                stub_log = build_stub_jsons_from_patches(folder)
                output_log += stub_log
                yield output_log, gr.update(interactive=False), SHOW_STOP

            for chunk in run_in_thread(Mothbot_Cluster.run, input_path=folder, dataset_root=dataset_root):
                output_log += chunk
                yield output_log, gr.update(interactive=False), SHOW_STOP
            output_log += f"✅ Cluster completed for {folder}\n"
        except Exception as exc:
            had_error = True
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP

    output_log += "------  Cluster  processing finished ------"
    yield output_log, gr.update(interactive=(not had_error)), HIDE_STOP


def run_cluster(selected_folders):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_Cluster.run,
        start_message="---🔍 Running Cluster for {folder} ---\n",
        success_message="✅  Cluster  completed for {folder}\n",
        finish_message="------  Cluster  processing finished ------",
        kwargs_builder=lambda folder: {"input_path": folder, "dataset_root": folder},
    )


def run_exif(selected_folders):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_InsertExif.run,
        start_message="---🔍 Running Insert Exif for {folder} ---\n",
        success_message="✅   Insert Exif completed for {folder}\n",
        finish_message="------  Insert Exif processing finished ------",
        kwargs_builder=lambda folder, dataset_root: {"input_path": folder, "dataset_root": dataset_root},
        skip_external=True,
    )


def run_full_process(
    selected_folders,
    yolo_model,
    imsz,
    overwrite_bot_detections,
    species_list,
    chosenrank,
    id_hum,
    id_bot,
    overwrite_bot_ids,
    metadata_csv,
    external_keys=None,
):
    SHOW_STOP = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP = gr.update(visible=False)
    if not selected_folders:
        yield "No image collections selected.\n", gr.update(visible=False)
        return

    # Steps that cannot run on externally-processed collections (no source images)
    source_only_steps = {"Detect", "Insert Exif"}

    steps = [
        (
            "Detect",
            Mothbot_Detect.run,
            lambda folder, dr: {
                "input_path": folder,
                "yolo_model": yolo_model,
                "imgsz": int(imsz),
                "overwrite_prev_bot_detections": bool(overwrite_bot_detections),
                "dataset_root": dr,
            },
        ),
        (
            "Cluster",
            Mothbot_Cluster.run,
            lambda folder, dr: {"input_path": folder, "dataset_root": dr},
        ),
        (
            "ID",
            Mothbot_ID.run,
            lambda folder, dr: {
                "input_path": folder,
                "taxa_csv": species_list,
                "rank": int(chosenrank),
                "ID_Hum": bool(id_hum),
                "ID_Bot": bool(id_bot),
                "overwrite_prev_bot_ID": bool(overwrite_bot_ids),
                "dataset_root": dr,
            },
        ),
        (
            "Insert Metadata",
            Mothbot_InsertMetadata.run,
            lambda folder, dr: {
                "input_path": folder,
                "metadata_path": str(metadata_csv),
                "dataset_root": dr,
            },
        ),
        (
            "Exif",
            Mothbot_InsertExif.run,
            lambda folder, dr: {"input_path": folder, "dataset_root": dr},
        ),
    ]

    output_log = ""
    for step_name, runner, kwargs_builder in steps:
        output_log += f"\n===== {step_name} =====\n"
        yield output_log, SHOW_STOP
        for entry in selected_folders:
            folder       = entry["path"]                     if isinstance(entry, dict) else entry
            is_ext       = entry.get("external", False)       if isinstance(entry, dict) else False
            dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

            if is_ext and step_name in source_only_steps:
                output_log += f"⚠️  Skipping {step_name} for externally-processed collection:\n    {folder}\n"
                yield output_log, SHOW_STOP
                continue

            if is_ext and step_name == "Cluster":
                output_log += f"  ℹ️  Building stub JSONs from patches before clustering {folder}...\n"
                yield output_log, SHOW_STOP
                output_log += build_stub_jsons_from_patches(folder)
                yield output_log, SHOW_STOP

            output_log += f"--- Running {step_name} for {folder} ---\n"
            yield output_log, SHOW_STOP
            try:
                for chunk in run_in_thread(runner, **kwargs_builder(folder, dataset_root)):
                    output_log += chunk
                    yield output_log, SHOW_STOP
                output_log += f"✅ {step_name} completed for {folder}\n"
            except Exception as exc:
                output_log += f"\n❌ Exception while processing {folder} in {step_name}: {exc}\n"
            yield output_log, SHOW_STOP

    output_log += "\n------ Full processing finished ------"
    yield output_log, HIDE_STOP


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def build_stub_jsons_from_patches(processed_folder):
    """For an externally-processed collection, reverse-build minimal stub JSON
    detection files from the patch images present in *processed_folder*.

    Each .jpg in *processed_folder* is treated as a patch.  The function groups
    patches by their source image stem (the filename minus the last two
    ``_<detidx>_<modelname>`` components) and writes one stub JSON per inferred
    source image, listing each patch as a shape with an empty detection record.

    This allows Cluster and ID to run on the collection even though no source
    images or original JSON files exist.

    Returns a log string describing what was created.
    """
    import json as _json
    from pathlib import Path as _Path
    import re as _re

    log = ""
    processed_folder = _Path(processed_folder)
    patch_files = sorted(processed_folder.glob("*.jpg"))

    # Group patches by inferred source stem.
    # Patch filename format: <source_stem>_<detidx>_<modelname>.jpg
    # We strip the last two "_"-separated components to recover the source stem.
    groups = {}
    for pf in patch_files:
        parts = pf.stem.rsplit("_", 2)
        if len(parts) >= 3:
            source_stem = "_".join(parts[:-2])
        else:
            source_stem = pf.stem  # can't parse — treat as its own group
        groups.setdefault(source_stem, []).append(pf)

    created = 0
    skipped = 0
    for source_stem, patches in groups.items():
        json_path = processed_folder / f"{source_stem}_botdetection.json"
        if json_path.exists():
            skipped += 1
            continue

        shapes = []
        for pf in sorted(patches):
            shapes.append({
                "label": "creature",
                "points": [],
                "patch_path": pf.name,
                "confidence_detection": None,
                "identifier_bot": "",
                "identifier_human": "",
                "timestamp_detection": "",
                "detector_bot": "external",
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "score": None,
                "direction": 0,
                "group_id": None,
                "description": "",
                "difficult": "false",
                "kie_linking": [],
            })

        stub = {
            "version": "external",
            "flags": {},
            "imagePath": source_stem + ".jpg",
            "imageHeight": None,
            "imageWidth": None,
            "description": "stub generated from external patches",
            "imageData": None,
            "shapes": shapes,
        }

        with open(json_path, "w") as fh:
            _json.dump(stub, fh, indent=4)
        created += 1

    log += f"  Stub JSON creation: {created} created, {skipped} already existed.\n"
    return log


def find_image_collections(directory, processed_dir_name="_processed"):
    """Walk *directory* and return every sub-folder (or *directory* itself)
    that contains at least one .jpg file, skipping the _processed tree and
    any patches/ folders.

    Returns a sorted list of absolute folder paths.
    """
    directory = os.path.abspath(directory)
    matches = []
    for root, dirs, files in os.walk(directory):
        # Prune the _processed tree and patches folders from the walk
        dirs[:] = sorted(
            d for d in dirs
            if d != processed_dir_name and d.lower() != "patches"
        )
        if any(f.lower().endswith(".jpg") for f in files):
            matches.append(os.path.abspath(root))
    return sorted(matches)


def _resolve_optional_path(*candidates):
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path.resolve())
    if candidates:
        return str(Path(candidates[0]).resolve())
    return ""


def _resolve_artifact_path(*candidates):
    return _resolve_optional_path(
        *[ARTIFACTS_DIR / candidate for candidate in candidates]
    )


def _resolve_first_artifact_match(pattern, fallback):
    matches = sorted(ARTIFACTS_DIR.glob(pattern))
    if matches:
        return str(matches[0].resolve())
    return _resolve_optional_path(ARTIFACTS_DIR / fallback)


def _browse_file(current_path, filetypes):
    return (
        browse_path(current_path=current_path, mode="file", filetypes=filetypes)
        or current_path
    )


def _count_matching_files(directory_path, patterns):
    return sum(
        len(glob.glob(os.path.join(directory_path, pattern))) for pattern in patterns
    )


def _run_batch_pipeline(
    selected_folders,
    runner,
    start_message,
    success_message,
    finish_message,
    kwargs_builder,
    skip_external=False,
):
    SHOW_STOP = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP = gr.update(visible=False)
    if not selected_folders:
        yield "No image collections selected.\n", gr.update(visible=False)
        return

    output_log = ""
    for entry in selected_folders:
        folder       = entry["path"]                     if isinstance(entry, dict) else entry
        is_ext       = entry.get("external", False)       if isinstance(entry, dict) else False
        dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

        if skip_external and is_ext:
            output_log += f"⚠️  Skipping (not applicable for externally-processed collection):\n    {folder}\n"
            yield output_log, SHOW_STOP
            continue

        output_log += start_message.format(folder=folder)
        yield output_log, SHOW_STOP

        try:
            for chunk in run_in_thread(runner, **kwargs_builder(folder, dataset_root)):
                output_log += chunk
                yield output_log, SHOW_STOP
            output_log += success_message.format(folder=folder)
        except Exception as exc:
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log, SHOW_STOP

    output_log += finish_message
    yield output_log, HIDE_STOP

'''
DEFAULT_METADATA_CSV = _resolve_artifact_path(
    "metadata.csv",
    Path("../artifacts/metadata.csv"),
    Path("defaults/metadata.csv"),
    Path("assets/metadata.csv"),
)
DEFAULT_SPECIES_CSV = _resolve_first_artifact_match(
    "species_list/*.csv",
    "species_list/species.csv",
)
DEFAULT_YOLO_MODEL = _resolve_first_artifact_match(
    "models/**/*.pt",
    "models/model.pt",
)
'''

# Temporarily disable the artifact matching to just leave the defaults blank #because i don't know how to use the artifact thing right yet

DEFAULT_METADATA_CSV = ""
DEFAULT_SPECIES_CSV = ""
DEFAULT_YOLO_MODEL = ""

demo = app()

if __name__ == "__main__":
    launch_kwargs = {"inbrowser": True}
    favicon = Path(__file__).with_name("favicon.png")
    if favicon.exists():
        launch_kwargs["favicon_path"] = str(favicon)
    ensure_single_instance(url="http://127.0.0.1:7860")
    start_tray(url="http://127.0.0.1:7860")
    demo.launch(**launch_kwargs)