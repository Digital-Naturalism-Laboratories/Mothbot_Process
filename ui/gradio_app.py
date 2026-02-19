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
from pathlib import Path
import gradio as gr

from core.common import run_in_thread
from ui.path_picker import browse_path

# Lazy-import worker modules so heavy ML deps only load when a tab is used.
from pipeline import cluster as Mothbot_Cluster
from pipeline import detect as Mothbot_Detect
from pipeline import identify as Mothbot_ID
from pipeline import insert_exif as Mothbot_InsertExif
from pipeline import insert_metadata as Mothbot_InsertMetadata

NIGHTLY_REGEX = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|[A-Za-z0-9]+_\d{4}-\d{2}-\d{2})$")
TAXA_COLS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(
    os.getenv("MOTHBOT_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
)


# ──────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────


def app():
    with gr.Blocks(
        title="Mothbot",
        css="""
            /* Tab 1 - Pastel Red */
            button.svelte-1ipelgc:nth-child(1).selected {
                background-color: #ff9999 !important;
                color: #ffffff !important;
            }
            /* Tab 2 - Pastel Orange */
            button.svelte-1ipelgc:nth-child(2).selected {
                background-color: #ffcc99 !important;
                color: #000000 !important;
            }
            /* Tab 3 - Pastel Yellow */
            button.svelte-1ipelgc:nth-child(3).selected {
                background-color: #ffff99 !important;
                color: #000000 !important;
            }
            /* Tab 4 - Pastel Green */
            button.svelte-1ipelgc:nth-child(4).selected {
                background-color: #ccffcc !important;
                color: #000000 !important;
            }
            /* Tab 5 - Pastel Blue */
            button.svelte-1ipelgc:nth-child(5).selected {
                background-color: #99ccff !important;
                color: #000000 !important;
            }
            /* Tab 6 - Pastel Indigo */
            button.svelte-1ipelgc:nth-child(6).selected {
                background-color: #cc99ff !important;
                color: #ffffff !important;
            }
        """,
    ) as demo:
        mapping_state = gr.State({})
        toggle_label_state = gr.State("Select All")
        selected_paths = gr.JSON(
            label="Confirmed Nightly Folders to be Processed", visible=False
        )

        with gr.Tabs(selected="setup") as main_tabs:
            # ~~~~~~~~~~~~ Setup TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Setup", id="setup"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "### Pick a main folder of Deployments to process: "
                        )
                        deployment_path = gr.Text(
                            label="Deployment Folder Path (paste or type)",
                            placeholder="/path/to/your/deployment/folder",
                            interactive=True,
                        )
                        deployment_browse_btn = gr.Button(
                            "Pick a Deployment Folder", size="sm", variant="primary"
                        )
                        with gr.Group():
                            status = gr.Textbox(
                                label="Error", lines=3, interactive=False, visible=False
                            )
                            folder_choices = gr.CheckboxGroup(
                                label="Nightly Folders (Select at least one night folder)",
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
                            with gr.Column():
                                metadata_csv_file = gr.Text(
                                    label="metadata field sheet:",
                                    value=DEFAULT_METADATA_CSV,
                                )
                                metadata_browse_btn = gr.Button("Browse", size="sm")
                            with gr.Column():
                                species_path = gr.Text(
                                    label="Species List:",
                                    value=DEFAULT_SPECIES_CSV,
                                )
                                species_browse_btn = gr.Button("Browse", size="sm")
                            with gr.Column():
                                yolo_model_path = gr.Text(
                                    value=DEFAULT_YOLO_MODEL,
                                    label="YOLO Model Path",
                                )
                                yolo_browse_btn = gr.Button("Browse", size="sm")
                        advanced_mode = gr.Checkbox(
                            label="Advanced mode",
                            value=False,
                        )

                deployment_browse_btn.click(
                    fn=browse_deployment_folder,
                    inputs=[deployment_path],
                    outputs=[deployment_path],
                ).then(
                    fn=scan_deployment_folder,
                    inputs=[deployment_path],
                    outputs=[
                        status,
                        folder_choices,
                        mapping_state,
                        toggle_label_state,
                        continue_process_btn,
                        selected_paths,
                        toggle_all_btn,
                    ],
                )
                deployment_path.change(
                    fn=scan_deployment_folder,
                    inputs=[deployment_path],
                    outputs=[
                        status,
                        folder_choices,
                        mapping_state,
                        toggle_label_state,
                        continue_process_btn,
                        selected_paths,
                        toggle_all_btn,
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
                    inputs=[folder_choices, mapping_state],
                    outputs=[selected_paths, continue_process_btn],
                )
                toggle_label_state.change(
                    lambda lbl: gr.update(value=lbl),
                    inputs=toggle_label_state,
                    outputs=toggle_all_btn,
                )
                folder_choices.change(
                    fn=confirm_selection,
                    inputs=[folder_choices, mapping_state],
                    outputs=[selected_paths, continue_process_btn],
                )
            # ~~~~~~~~~~~~ PROCESS TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Process", id="process"):
                process_output_box = gr.Textbox(
                    label="Process Output", lines=20, interactive=False
                )

            # ~~~~~~~~~~~~ DETECTION TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Detect", id="detect", visible=False) as detect_tab:
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
                continue_id_btn = gr.Button(
                    "Continue to ID", variant="primary", interactive=False
                )

                DET_run_btn.click(
                    fn=run_detection_with_continue,
                    inputs=[
                        selected_paths,
                        yolo_model_path,
                        imgsz,
                        OVERWRITE_PREV_BOT_DETECTIONS,
                    ],
                    outputs=[DET_output_box, continue_id_btn],
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

                ID_run_btn = gr.Button("Run Identification", variant="primary")
                ID_output_box = gr.Textbox(label="Identification Output", lines=20)

                ID_run_btn.click(
                    fn=run_ID,
                    inputs=[
                        selected_paths,
                        species_path,
                        taxa_output,
                        ID_HUMANDETECTIONS,
                        ID_BOTDETECTIONS,
                        OVERWRITE_PREV_BOT_IDENTIFICATIONS,
                    ],
                    outputs=ID_output_box,
                )

            # ~~~~~~~~~~~~ Metadata Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Insert Metadata", id="metadata", visible=False) as metadata_tab:
                metadata_run_btn = gr.Button("Insert Metadata", variant="primary")
                metadata_output_box = gr.Textbox(
                    label="Insert Metadata Output", lines=20
                )

                metadata_run_btn.click(
                    fn=run_metadata,
                    inputs=[selected_paths, metadata_csv_file],
                    outputs=metadata_output_box,
                )

            # ~~~~~~~~~~~~ Cluster Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Cluster Perceptually", id="cluster", visible=False) as cluster_tab:
                cluster_run_btn = gr.Button("Cluster Perceptually", variant="primary")
                cluster_output_box = gr.Textbox(label="Cluster Output", lines=20)

                cluster_run_btn.click(
                    fn=run_cluster,
                    inputs=[selected_paths],
                    outputs=cluster_output_box,
                )

            # ~~~~~~~~~~~~ Exif Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Insert Exif", id="exif", visible=False) as exif_tab:
                exif_run_btn = gr.Button("Insert Exif (Optional)", variant="primary")
                exif_output_box = gr.Textbox(label="Insert Exif Output", lines=20)

                exif_run_btn.click(
                    fn=run_exif,
                    inputs=[selected_paths],
                    outputs=exif_output_box,
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
                outputs=process_output_box,
            )

    return demo


# ──────────────────────────────────────────────────────────────
#  Functions called by the UI
# ──────────────────────────────────────────────────────────────


def browse_deployment_folder(current_path):
    return browse_path(current_path=current_path, mode="folder") or current_path


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


def scan_deployment_folder(folder_path):
    """Scan *folder_path* for nightly sub-folders and return UI updates."""
    if not folder_path or not os.path.isdir(folder_path):
        return (
            gr.update(value="No valid folder path provided.", visible=True),
            gr.update(choices=[], value=[]),
            {},
            "Select All",
            gr.update(interactive=False, visible=False),
            [],
            gr.update(visible=False),
        )

    matches = find_nightly_folders_recursive(folder_path)
    if not matches:
        return (
            gr.update(
                value=f"No nightly subfolders found in:\n{folder_path}", visible=True
            ),
            gr.update(choices=[], value=[]),
            {},
            "Select All",
            gr.update(interactive=False, visible=False),
            [],
            gr.update(visible=False),
        )

    choices = []
    mapping = {}
    seen_values = set()

    for p in matches:
        base_value = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
        value = base_value
        i = 1
        while value in seen_values:
            value = f"{base_value} ({i})"
            i += 1
        seen_values.add(value)

        jpeg_count = _count_matching_files(p, ("*.jpg", "*.jpeg"))
        json_count = _count_matching_files(p, ("*.json",))
        patches_folder = os.path.join(p, "patches")
        patches_count = 0
        if os.path.isdir(patches_folder):
            patches_count = _count_matching_files(patches_folder, ("*.jpg", "*.jpeg"))

        decorated_label = f"{value} ({jpeg_count} Images, {json_count} JSONs, {patches_count} Patches)"
        choices.append((decorated_label, value))
        mapping[value] = os.path.abspath(p)

    status = f"Selected folder: {folder_path}\nFound {len(choices)} nightly folders."
    return (
        gr.update(value="", visible=False),
        gr.update(choices=choices, value=[], visible=True),
        mapping,
        "Select All",
        gr.update(interactive=False, visible=True),
        [],
        gr.update(visible=True),
    )


def toggle_select_all(current_values, mapping, button_label):
    del current_values
    if button_label == "Select All":
        return gr.update(value=list(mapping.keys())), "Deselect All"
    return gr.update(value=[]), "Select All"


def confirm_selection(selected_labels, mapping):
    if not selected_labels:
        return [], gr.update(interactive=False)
    resolved = [mapping[label] for label in selected_labels if label in mapping]
    return resolved, gr.update(interactive=bool(resolved))


def go_to_process_tab():
    return gr.Tabs(selected="process")


def go_to_id_tab():
    return gr.Tabs(selected="id")


def toggle_advanced_mode(enabled):
    visible = bool(enabled)
    selected_tab = "setup" if visible else "process"
    return (
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.Tabs(selected=selected_tab),
    )


def get_index(selected_word):
    return TAXA_COLS.index(selected_word)


def run_detection_with_continue(selected_folders, yolo_model, imsz, overwrite_bot):
    if not selected_folders:
        yield "No nightly folders selected.\n", gr.update(interactive=False)
        return

    output_log = ""
    had_error = False

    for folder in selected_folders:
        output_log += f"---🕵🏾‍♀️ Running detection for {folder} ---\n"
        yield output_log, gr.update(interactive=False)

        try:
            for chunk in run_in_thread(
                Mothbot_Detect.run,
                input_path=folder,
                yolo_model=yolo_model,
                imgsz=int(imsz),
                overwrite_prev_bot_detections=bool(overwrite_bot),
            ):
                output_log += chunk
                yield output_log, gr.update(interactive=False)
            output_log += f"✅ Detection completed for {folder}\n"
        except Exception as exc:
            had_error = True
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log, gr.update(interactive=False)

    output_log += "----------- Finished running Batch --------------"
    yield output_log, gr.update(interactive=(not had_error))


def run_ID(selected_folders, species_list, chosenrank, IDHum, IDBot, overwrite_bot):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_ID.run,
        start_message="---🔍 Running IDENTIFICATION for {folder} ---\n",
        success_message="✅ Identification completed for {folder}\n",
        finish_message="------ ID processing finished ------",
        kwargs_builder=lambda folder: {
            "input_path": folder,
            "taxa_csv": species_list,
            "rank": int(chosenrank),
            "ID_Hum": bool(IDHum),
            "ID_Bot": bool(IDBot),
            "overwrite_prev_bot_ID": bool(overwrite_bot),
        },
    )


def run_metadata(selected_folders, metadata):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_InsertMetadata.run,
        start_message="---🔍 Running METADATA for {folder} ---\n",
        success_message="✅ Insert Metadata completed for {folder}\n",
        finish_message="------ Insert Metadata processing finished ------",
        kwargs_builder=lambda folder: {
            "input_path": folder,
            "metadata_path": str(metadata),
        },
    )


def run_cluster(selected_folders):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_Cluster.run,
        start_message="---🔍 Running Cluster for {folder} ---\n",
        success_message="✅  Cluster  completed for {folder}\n",
        finish_message="------  Cluster  processing finished ------",
        kwargs_builder=lambda folder: {"input_path": folder},
    )


def run_exif(selected_folders):
    yield from _run_batch_pipeline(
        selected_folders=selected_folders,
        runner=Mothbot_InsertExif.run,
        start_message="---🔍 Running Insert Exif for {folder} ---\n",
        success_message="✅   Insert Exif completed for {folder}\n",
        finish_message="------  Insert Exif processing finished ------",
        kwargs_builder=lambda folder: {"input_path": folder},
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
):
    if not selected_folders:
        yield "No nightly folders selected.\n"
        return

    steps = [
        (
            "Detect",
            Mothbot_Detect.run,
            lambda folder: {
                "input_path": folder,
                "yolo_model": yolo_model,
                "imgsz": int(imsz),
                "overwrite_prev_bot_detections": bool(overwrite_bot_detections),
            },
        ),
        (
            "ID",
            Mothbot_ID.run,
            lambda folder: {
                "input_path": folder,
                "taxa_csv": species_list,
                "rank": int(chosenrank),
                "ID_Hum": bool(id_hum),
                "ID_Bot": bool(id_bot),
                "overwrite_prev_bot_ID": bool(overwrite_bot_ids),
            },
        ),
        (
            "Insert Metadata",
            Mothbot_InsertMetadata.run,
            lambda folder: {
                "input_path": folder,
                "metadata_path": str(metadata_csv),
            },
        ),
        (
            "Cluster",
            Mothbot_Cluster.run,
            lambda folder: {"input_path": folder},
        ),
        (
            "Exif",
            Mothbot_InsertExif.run,
            lambda folder: {"input_path": folder},
        ),
    ]

    output_log = ""
    for step_name, runner, kwargs_builder in steps:
        output_log += f"\n===== {step_name} =====\n"
        yield output_log
        for folder in selected_folders:
            output_log += f"--- Running {step_name} for {folder} ---\n"
            yield output_log
            try:
                for chunk in run_in_thread(runner, **kwargs_builder(folder)):
                    output_log += chunk
                    yield output_log
                output_log += f"✅ {step_name} completed for {folder}\n"
            except Exception as exc:
                output_log += f"\n❌ Exception while processing {folder} in {step_name}: {exc}\n"
            yield output_log

    output_log += "\n------ Full processing finished ------"
    yield output_log


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def find_nightly_folders_recursive(directory):
    matches = []
    if NIGHTLY_REGEX.match(os.path.basename(directory)):
        matches.append(os.path.abspath(directory))
    for root, dirs, _ in os.walk(directory):
        for folder_name in dirs:
            if NIGHTLY_REGEX.match(folder_name):
                matches.append(os.path.join(root, folder_name))
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
):
    if not selected_folders:
        yield "No nightly folders selected.\n"
        return

    output_log = ""
    for folder in selected_folders:
        output_log += start_message.format(folder=folder)
        yield output_log

        try:
            for chunk in run_in_thread(runner, **kwargs_builder(folder)):
                output_log += chunk
                yield output_log
            output_log += success_message.format(folder=folder)
        except Exception as exc:
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log

    output_log += finish_message
    yield output_log


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

demo = app()

if __name__ == "__main__":
    launch_kwargs = {"inbrowser": True}
    favicon = Path(__file__).with_name("favicon.png")
    if favicon.exists():
        launch_kwargs["favicon_path"] = str(favicon)
    demo.launch(**launch_kwargs)
