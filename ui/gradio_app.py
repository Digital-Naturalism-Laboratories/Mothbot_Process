#!/usr/bin/env python3
"""
Mothbot Gradio UI – desktop-packaging-friendly version.

Key changes from the subprocess-based original:
  * Worker scripts are called via their ``run()`` functions (in-process).
  * stdout is captured via ``core.common.run_in_thread`` and streamed into
    Gradio Textbox outputs — same UX, no subprocess overhead.
  * Path fields support both paste/type and optional native browse dialogs.
"""

import json
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

from core.common import run_in_thread, request_cancel, find_images_recursive
from core.preview import get_preview, clear_preview, emit_preview
from ui.tray import start_tray
from ui.single_instance import ensure_single_instance
from ui.path_picker import browse_path, browse_path_with_status

# Lazy-import worker modules so heavy ML deps only load when a tab is used.
from pipeline import cluster as Mothbot_Cluster
from pipeline import detect as Mothbot_Detect
from pipeline import identify as Mothbot_ID
from pipeline import insert_exif as Mothbot_InsertExif
from pipeline import insert_metadata as Mothbot_InsertMetadata
from pipeline import pixel_mass as Mothbot_PixelMass

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
        js="""
        function() {
            // Tag the Pixel Mass tab button with a data attribute so CSS can color
            // it without relying on nth-child counts (which shift when tabs are
            // shown/hidden).  Re-run on DOM mutations so it survives Gradio re-renders.
            function tagPixelMassTab() {
                var tabs = document.querySelectorAll('#mothbot-tabs button');
                tabs.forEach(function(btn) {
                    if (btn.textContent.trim() === 'Pixel Mass') {
                        btn.setAttribute('data-tab', 'pixel-mass');
                    }
                });
            }
            tagPixelMassTab();
            new MutationObserver(tagPixelMassTab).observe(document.body, { childList: true, subtree: true });
        }
        """,
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
            /* Pixel Mass tab - orange. Targeted by data attribute set via JS below
               so the color is immune to nth-child counting shifts. */
            #mothbot-tabs button[data-tab="pixel-mass"]:not(.selected) {
                border-bottom: 3px solid #ff8c00 !important;
            }
            #mothbot-tabs button[data-tab="pixel-mass"].selected {
                background-color: #ff8c00 !important;
                color: #ffffff !important;
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

        with gr.Tabs(selected="setup", elem_id="mothbot-tabs") as main_tabs:
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
                with gr.Row():
                    process_output_box = gr.Textbox(
                        label="Process Output", lines=20, interactive=False, scale=2
                    )
                    process_preview_img = gr.Image(
                        label="Live Detection Preview (every 10th image)",
                        visible=False,
                        scale=1,
                        show_download_button=False,
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
                with gr.Row():
                    DET_output_box = gr.Textbox(label="Detection Output", lines=20, scale=2)
                    DET_preview_img = gr.Image(
                        label="Live Detection Preview (every 10th image)",
                        visible=False,
                        scale=1,
                        show_download_button=False,
                    )

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
                    outputs=[DET_output_box, continue_cluster_btn, stop_btn, DET_preview_img],
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

            # ~~~~~~~~~~~~ Pixel Mass Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Pixel Mass", id="pixel_mass", visible=False) as pixel_mass_tab:
                pm_source_img_state = gr.State(None)   # thumbnail PIL for fast redrawing
                pm_point1_state     = gr.State(None)   # [x, y] of first click (thumbnail space)
                pm_point2_state     = gr.State(None)   # [x, y] of second click (thumbnail space)
                pm_scale_state      = gr.State(1.0)    # thumbnail / original ratio (for px/mm conversion)

                with gr.Accordion("Step 1: Set Scale Calibration", open=True) as pm_step1_accordion:
                    gr.Markdown(
                        "Click two points on a ruler or known object in the source image below. "
                        "Enter the real-world distance and click **Apply Calibration**. "
                        "Or type a known **pixels per mm** value directly and apply."
                    )
                    with gr.Row():
                        pm_calib_img = gr.Image(
                            label="Source image — click two points to mark a known distance",
                            interactive=False,
                            scale=2,
                        )
                        with gr.Column(scale=1):
                            pm_load_img_btn = gr.Button("Load Different Image", size="sm")
                            pm_point1_label = gr.Textbox(
                                label="Point 1", value="–", interactive=False, lines=1, max_lines=1
                            )
                            pm_point2_label = gr.Textbox(
                                label="Point 2", value="–", interactive=False, lines=1, max_lines=1
                            )
                            pm_pixel_dist_label = gr.Textbox(
                                label="Pixel distance", value="–", interactive=False, lines=1, max_lines=1
                            )
                            pm_real_dist = gr.Number(label="Real-world distance (mm)", value=10.0, minimum=0.001)
                            pm_pixels_per_mm = gr.Number(label="Pixels per mm (auto-computed or enter manually)")
                            pm_calibrate_btn = gr.Button("Apply Calibration", variant="primary")
                            pm_calib_status = gr.Textbox(
                                label="Calibration status", value="", interactive=False, lines=1, max_lines=1
                            )

                gr.Markdown("### Step 2: Calculate Pixel Mass")
                with gr.Row():
                    pm_overwrite_nobg = gr.Checkbox(
                        label="Overwrite previous transparent images", value=False
                    )
                    pm_overwrite_pixmass = gr.Checkbox(
                        label="Overwrite previous pixel mass", value=True
                    )
                pm_run_btn = gr.Button("Run Pixel Mass", variant="primary")
                with gr.Row():
                    pm_output_box = gr.Textbox(
                        label="Pixel Mass Output", lines=15, interactive=False, scale=2
                    )
                    pm_preview_img = gr.Image(
                        label="Latest patch (bg removed)", interactive=False, scale=1
                    )

                # ── Calibration event handlers ──────────────────────────────
                # Auto-load source image when this tab is opened.
                _pm_load_outputs = [
                    pm_calib_img, pm_source_img_state, pm_calib_status,
                    pm_point1_state, pm_point2_state,
                    pm_point1_label, pm_point2_label, pm_pixel_dist_label,
                    pm_scale_state,
                ]
                pixel_mass_tab.select(
                    fn=load_image_for_calibration,
                    inputs=[selected_paths],
                    outputs=_pm_load_outputs,
                )

                pm_load_img_btn.click(
                    fn=load_image_for_calibration,
                    inputs=[selected_paths],
                    outputs=_pm_load_outputs,
                )

                pm_calib_img.select(
                    fn=mark_calibration_point,
                    inputs=[pm_point1_state, pm_point2_state, pm_source_img_state],
                    outputs=[pm_point1_state, pm_point2_state, pm_calib_img,
                             pm_point1_label, pm_point2_label, pm_pixel_dist_label],
                )

                pm_calibrate_btn.click(
                    fn=apply_calibration,
                    inputs=[selected_paths, pm_point1_state, pm_point2_state,
                            pm_real_dist, pm_pixels_per_mm, pm_scale_state],
                    outputs=[pm_pixels_per_mm, pm_calib_status],
                )

                pm_run_btn.click(
                    fn=run_pixel_mass_ui,
                    inputs=[selected_paths, pm_pixels_per_mm, pm_overwrite_nobg, pm_overwrite_pixmass],
                    outputs=[pm_output_box, stop_btn, pm_step1_accordion, pm_preview_img],
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
                    pixel_mass_tab,
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
                outputs=[process_output_box, stop_btn, process_preview_img],
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


def _check_pipeline_status(processed_mirror: str) -> dict:
    """Sample one JSON from *processed_mirror* and return booleans for each
    pipeline stage that has already been run on this collection."""
    status = {"clustered": False, "identified": False, "metadata": False, "exif": False, "pixel_mass": False}
    if not os.path.isdir(processed_mirror):
        return status

    # Find first JSON; prefer one with shapes so cluster/ID checks are meaningful.
    first_json_path = None
    shaped_json_data = None
    for root, _dirs, files in os.walk(processed_mirror):
        for f in files:
            if not f.endswith(".json"):
                continue
            path = os.path.join(root, f)
            if first_json_path is None:
                first_json_path = path
            if shaped_json_data is None:
                try:
                    with open(path) as fh:
                        d = json.load(fh)
                    if d.get("shapes"):
                        shaped_json_data = d
                        break
                except Exception:
                    pass
        if shaped_json_data:
            break

    # Metadata: detect.py never writes "latitude", so its mere presence means
    # insert_metadata.py has been run.
    if first_json_path:
        try:
            with open(first_json_path) as fh:
                d = json.load(fh)
            status["metadata"] = "latitude" in d
        except Exception:
            pass

    if shaped_json_data:
        shapes = shaped_json_data.get("shapes", [])
        status["clustered"] = any(s.get("clusterID") is not None for s in shapes)
        # detect.py writes identifier_bot="" (empty); identify.py sets it to the
        # version string, so a non-empty value means identification has been run.
        status["identified"] = any(s.get("identifier_bot", "") not in ("", None) for s in shapes)
        status["pixel_mass"] = any("pixel_mass_pixels" in s for s in shapes)

    # Exif: check whether the first patch JPG in the mirror has GPS EXIF data.
    for root, _dirs, files in os.walk(processed_mirror):
        for f in files:
            if f.lower().endswith(".jpg"):
                try:
                    import piexif
                    exif_dict = piexif.load(os.path.join(root, f))
                    status["exif"] = bool(exif_dict.get("GPS"))
                except Exception:
                    pass
                return status
        break  # only peek one level deep for speed

    return status


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
            # For external collections the folder IS the processed mirror.
            json_count   = _count_matching_files(p, ("*.json",))
            patch_count  = jpeg_count
            ps = _check_pipeline_status(p)
            counts = f"📄 {json_count}  🦋 {patch_count}"
        else:
            processed_mirror = os.path.join(folder_path, "_processed", os.path.relpath(p, folder_path))
            json_count  = _count_matching_files(processed_mirror, ("*.json",))  if os.path.isdir(processed_mirror) else 0
            patch_count = _count_matching_files(processed_mirror, ("*.jpg", "*.jpeg")) if os.path.isdir(processed_mirror) else 0
            ps = _check_pipeline_status(processed_mirror)
            counts = f"📷 {jpeg_count}  📄 {json_count}  🦋 {patch_count}"

        pipeline_tags = "  ".join(
            tag for flag, tag in [
                (ps["clustered"],   "✓ Cluster"),
                (ps["identified"],  "✓ ID"),
                (ps["metadata"],    "✓ Meta"),
                (ps["exif"],        "✓ Exif"),
                (ps["pixel_mass"],  "✓ PixMass"),
            ]
            if flag
        )

        if is_external:
            label = f"⚡ {label_prefix}  {counts}  [ext]"
        else:
            label = f"{label_prefix}  {counts}"
        if pipeline_tags:
            label += f"  |  {pipeline_tags}"

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

_CALIB_MAX_PX = 1200   # longest edge of the working thumbnail


def load_image_for_calibration(selected_folders):
    """Load the first source image, downsample to a working thumbnail, return scale factor."""
    _RESET = ("–", "–", "–")
    _FAIL  = (None, None, "–", None, None, *_RESET, 1.0)
    if not selected_folders:
        return (None, None, "No collection selected.", *_FAIL[3:])
    entry = selected_folders[0]
    folder = entry["path"] if isinstance(entry, dict) else entry
    images = find_images_recursive(folder)
    if not images:
        return (None, None, "No source images found in collection.", *_FAIL[3:])
    from PIL import Image as PILImage
    img = PILImage.open(images[0]).convert("RGB")
    w, h = img.size
    scale = min(1.0, _CALIB_MAX_PX / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    status = f"Loaded: {os.path.basename(images[0])} ({w}×{h})"
    return img, img.copy(), status, None, None, *_RESET, scale


def mark_calibration_point(evt: gr.SelectData, p1, p2, orig_pil):
    """Cycle through first-click → second-click → reset on the calibration thumbnail."""
    import math
    from PIL import ImageDraw

    if orig_pil is None:
        return p1, p2, None, "–", "–", "–"

    x, y = int(evt.index[0]), int(evt.index[1])

    if p1 is None:
        new_p1, new_p2 = [x, y], None
    elif p2 is None:
        new_p1, new_p2 = p1, [x, y]
    else:
        new_p1, new_p2 = [x, y], None   # third click resets

    annotated = orig_pil.copy()
    draw = ImageDraw.Draw(annotated)
    r, lw = 6, 2   # small fixed-pixel markers on the ~1200px thumbnail

    if new_p1:
        px1, py1 = new_p1
        draw.ellipse([px1 - r, py1 - r, px1 + r, py1 + r], fill="red", outline="white", width=lw)

    px_dist_str = "–"
    if new_p2:
        px2, py2 = new_p2
        draw.line([new_p1[0], new_p1[1], px2, py2], fill="yellow", width=lw)
        draw.ellipse([px2 - r, py2 - r, px2 + r, py2 + r], fill="blue", outline="white", width=lw)
        dist = math.sqrt((new_p1[0] - px2) ** 2 + (new_p1[1] - py2) ** 2)
        px_dist_str = f"{dist:.1f} px (thumbnail)"

    p1_str = f"({new_p1[0]}, {new_p1[1]})" if new_p1 else "–"
    p2_str = f"({new_p2[0]}, {new_p2[1]})" if new_p2 else "–"

    return new_p1, new_p2, annotated, p1_str, p2_str, px_dist_str


def apply_calibration(selected_folders, p1, p2, real_dist_mm, manual_ppm, scale=1.0):
    """Compute pixels_per_mm from the marked line (or use manual value) and save calibration.json.

    Points p1/p2 are in thumbnail coordinate space; scale (thumbnail/original) converts
    the measured pixel distance back to original-image pixels before dividing by real_dist_mm.
    """
    import math
    from core.paths import get_processed_folder
    from core.common import current_timestamp
    from pipeline.pixel_mass import save_calibration

    ppm = None
    if p1 and p2 and real_dist_mm and real_dist_mm > 0:
        px_dist_thumb = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        px_dist_orig  = px_dist_thumb / (scale or 1.0)   # convert to original-image pixels
        ppm = px_dist_orig / real_dist_mm
    elif manual_ppm and manual_ppm > 0:
        ppm = float(manual_ppm)

    if ppm is None:
        return manual_ppm, "⚠️  Mark two points + enter distance, or type px/mm directly."

    if not selected_folders:
        return ppm, f"Computed {ppm:.4f} px/mm (no collection selected — not saved)"

    saved = 0
    for entry in selected_folders:
        folder = entry["path"] if isinstance(entry, dict) else entry
        dr = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder
        processed_folder = get_processed_folder(folder, dr)
        save_calibration(processed_folder, {
            "pixels_per_mm": ppm,
            "point1": p1,
            "point2": p2,
            "real_distance_mm": real_dist_mm,
            "calibration_date": current_timestamp(),
        })
        saved += 1

    return ppm, f"✅ {ppm:.4f} px/mm saved to {saved} collection(s)"


def _nobg_preview(path: str):
    """Composite a transparent _nobg.png onto a hot-pink/white checkerboard."""
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGBA")
    w, h = img.size
    tile = 16
    bg = PILImage.new("RGBA", (w, h))
    c1 = (255, 20, 147, 255)   # deep pink
    c2 = (255, 255, 255, 255)  # white
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            col = c1 if ((x // tile) + (y // tile)) % 2 == 0 else c2
            bw, bh = min(tile, w - x), min(tile, h - y)
            bg.paste(PILImage.new("RGBA", (bw, bh), col), (x, y))
    bg.paste(img, mask=img)
    return bg.convert("RGB")


def run_pixel_mass_ui(selected_folders, pixels_per_mm, overwrite_nobg, overwrite_pixmass):
    """Gradio generator that runs pixel_mass.run() for each selected collection."""
    SHOW_STOP  = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP  = gr.update(visible=False)
    COLLAPSE   = gr.update(open=False)
    EXPAND     = gr.update(open=True)
    NO_PREVIEW = gr.update()

    if not selected_folders:
        yield "No image collections selected.\n", HIDE_STOP, EXPAND, NO_PREVIEW
        return

    output_log = ""
    yield output_log, SHOW_STOP, COLLAPSE, NO_PREVIEW   # collapse Step 1 at start

    for entry in selected_folders:
        folder = entry["path"] if isinstance(entry, dict) else entry
        dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

        output_log += f"--- Pixel Mass for {folder} ---\n"
        yield output_log, SHOW_STOP, COLLAPSE, NO_PREVIEW
        try:
            for chunk in run_in_thread(
                Mothbot_PixelMass.run,
                input_path=folder,
                dataset_root=dataset_root,
                pixels_per_mm=float(pixels_per_mm) if pixels_per_mm else None,
                overwrite_nobg=bool(overwrite_nobg),
                overwrite_pixmass=bool(overwrite_pixmass),
            ):
                output_log += chunk
                preview_path = get_preview()
                if preview_path:
                    try:
                        preview_update = gr.update(value=_nobg_preview(preview_path))
                    except Exception:
                        preview_update = NO_PREVIEW
                else:
                    preview_update = NO_PREVIEW
                yield output_log, SHOW_STOP, COLLAPSE, preview_update
            output_log += f"✅ Pixel Mass completed for {folder}\n"
        except Exception as exc:
            output_log += f"\n❌ Exception: {exc}\n"
        yield output_log, SHOW_STOP, COLLAPSE, NO_PREVIEW

    output_log += "\n--- Pixel Mass finished ---"
    yield output_log, HIDE_STOP, EXPAND, NO_PREVIEW   # re-expand Step 1 when done


def toggle_advanced_mode(enabled):
    visible = bool(enabled)
    selected_tab = "setup"
    return (
        gr.update(visible=visible),      # detect_tab
        gr.update(visible=visible),      # id_tab
        gr.update(visible=visible),      # metadata_tab
        gr.update(visible=visible),      # cluster_tab
        gr.update(visible=visible),      # exif_tab
        gr.update(visible=visible),      # pixel_mass_tab
        gr.update(visible=not visible),  # process_tab (inverted — basic mode only)
        gr.Tabs(selected=selected_tab),
    )


def get_index(selected_word):
    return TAXA_COLS.index(selected_word)


def run_detection_with_continue(selected_folders, yolo_model, imsz, overwrite_bot, external_keys=None):
    SHOW_STOP = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP = gr.update(visible=False)
    NO_IMG = gr.update()  # no-op update for the preview image

    if not selected_folders:
        yield "No image collections selected.\n", gr.update(interactive=False), gr.update(visible=False), NO_IMG
        return

    clear_preview()
    external_keys = external_keys or set()
    output_log = ""
    had_error = False
    current_preview = None

    def _preview_update():
        nonlocal current_preview
        path = get_preview()
        if path:
            current_preview = path
            try:
                from PIL import Image as PILImage
                return gr.update(value=PILImage.open(path), visible=True)
            except Exception:
                return NO_IMG
        return NO_IMG

    for entry in selected_folders:
        folder       = entry["path"]                     if isinstance(entry, dict) else entry
        is_ext       = entry.get("external", False)       if isinstance(entry, dict) else False
        dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

        if is_ext:
            output_log += f"⚠️  Skipping detection for externally-processed collection (no source images):\n    {folder}\n"
            yield output_log, gr.update(interactive=False), SHOW_STOP, NO_IMG
            continue

        output_log += f"---🕵🏾‍♀️ Running detection for {folder} ---\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP, NO_IMG

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
                yield output_log, gr.update(interactive=False), SHOW_STOP, _preview_update()
            output_log += f"✅ Detection completed for {folder}\n"
        except Exception as exc:
            had_error = True
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP, NO_IMG

    output_log += "----------- Finished running Batch --------------"
    yield output_log, gr.update(interactive=(not had_error)), HIDE_STOP, NO_IMG


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

    clear_preview()
    output_log = ""
    NO_IMG = gr.update()

    def _poll_preview():
        path = get_preview()
        if path:
            try:
                from PIL import Image as PILImage
                return gr.update(value=PILImage.open(path), visible=True)
            except Exception:
                pass
        return NO_IMG

    for step_name, runner, kwargs_builder in steps:
        output_log += f"\n===== {step_name} =====\n"
        yield output_log, SHOW_STOP, NO_IMG
        for entry in selected_folders:
            folder       = entry["path"]                     if isinstance(entry, dict) else entry
            is_ext       = entry.get("external", False)       if isinstance(entry, dict) else False
            dataset_root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder

            if is_ext and step_name in source_only_steps:
                output_log += f"⚠️  Skipping {step_name} for externally-processed collection:\n    {folder}\n"
                yield output_log, SHOW_STOP, NO_IMG
                continue

            if is_ext and step_name == "Cluster":
                output_log += f"  ℹ️  Building stub JSONs from patches before clustering {folder}...\n"
                yield output_log, SHOW_STOP, NO_IMG
                output_log += build_stub_jsons_from_patches(folder)
                yield output_log, SHOW_STOP, NO_IMG

            output_log += f"--- Running {step_name} for {folder} ---\n"
            yield output_log, SHOW_STOP, NO_IMG
            try:
                for chunk in run_in_thread(runner, **kwargs_builder(folder, dataset_root)):
                    output_log += chunk
                    img = _poll_preview() if step_name == "Detect" else NO_IMG
                    yield output_log, SHOW_STOP, img
                output_log += f"✅ {step_name} completed for {folder}\n"
            except Exception as exc:
                output_log += f"\n❌ Exception while processing {folder} in {step_name}: {exc}\n"
            yield output_log, SHOW_STOP, NO_IMG

    output_log += "\n------ Full processing finished ------"
    yield output_log, HIDE_STOP, NO_IMG


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