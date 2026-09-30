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
import cv2
import sys
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import tomllib
import gradio as gr

from core.common import run_in_thread, request_cancel, find_images_recursive, TICK
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
from pipeline import legacy_converter as Mothbot_LegacyConverter
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

            // ── Collection labels: bold folder name, smaller "earlier" runs ─────
            // Checkbox labels are plain text in Gradio, so format a copy beside
            // Gradio's own text (kept, hidden) rather than editing it: Gradio keeps
            // updating its text node when choices change, and the copy follows.
            function _escape(t) {
                return t.replace(/[&<>"]/g, function(c) {
                    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c];
                });
            }
            function _formatCollectionLabel(text) {
                var parts = text.split('  |  ');
                var head = parts[0];
                var gap = head.indexOf('  ');
                var path = gap >= 0 ? head.slice(0, gap) : head;
                var rest = gap >= 0 ? head.slice(gap) : '';
                var slash = path.lastIndexOf('/');
                var html = _escape(path.slice(0, slash + 1)) + '<b>' + _escape(path.slice(slash + 1)) + '</b>' + _escape(rest);
                for (var i = 1; i < parts.length; i++) {
                    var part = _escape(parts[i]);
                    html += '  |  ' + (parts[i].indexOf('earlier ') === 0
                        ? '<span class="mb-earlier-run">' + part + '</span>' : part);
                }
                return html;
            }
            function formatCollectionLabels() {
                var spans = document.querySelectorAll('#collection-choices label > span:not(.mb-formatted-label)');
                spans.forEach(function(span) {
                    var text = span.textContent;
                    var copy = span.nextElementSibling;
                    if (!copy || !copy.classList.contains('mb-formatted-label')) {
                        copy = document.createElement('span');
                        copy.className = span.className + ' mb-formatted-label';
                        span.after(copy);
                        span.classList.add('mb-source-label');
                    }
                    if (copy.dataset.source !== text) {
                        copy.dataset.source = text;
                        copy.innerHTML = _formatCollectionLabel(text);
                    }
                });
            }
            formatCollectionLabels();
            new MutationObserver(formatCollectionLabels).observe(document.body, { childList: true, subtree: true, characterData: true });

            // ── Pixel Mass calibration viewer ──────────────────────────────────
            // Zoom / pan / point placement all happen here in the browser on the
            // full-resolution image. Points (original-image pixels) are reported
            // to Python by writing JSON into the hidden #pm-calib-points textbox.
            function initCalibViewer() {
                var root = document.getElementById('pm-calib-viewer');
                if (!root || root.dataset.ready === root.dataset.src) return;
                root.dataset.ready = root.dataset.src;
                var canvas = root.querySelector('canvas');
                var ctx = canvas.getContext('2d');
                var img = new Image();
                var pts = [], s = 1, tx = 0, ty = 0, fitS = 1, fitted = false;
                var cw = 0, ch = 0, drag = null;
                var MAX_ZOOM = 64, HIT_PX = 12, CLICK_SLOP = 4;
                var COLORS = ['#ff3b30', '#2f7bff'];

                function report() {
                    var box = document.querySelector('#pm-calib-points textarea, #pm-calib-points input');
                    if (!box) return;
                    box.value = JSON.stringify(pts.map(function(p) {
                        return [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100];
                    }));
                    box.dispatchEvent(new Event('input', { bubbles: true }));
                }
                function resize() {
                    if (!img.naturalWidth || !root.clientWidth) return;
                    var dpr = window.devicePixelRatio || 1;
                    cw = root.clientWidth;
                    ch = Math.min(cw * img.naturalHeight / img.naturalWidth, window.innerHeight * 0.75);
                    canvas.style.height = ch + 'px';
                    canvas.width = Math.round(cw * dpr);
                    canvas.height = Math.round(ch * dpr);
                    fitS = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
                    if (!fitted) { fit(); fitted = true; } else { draw(); }
                }
                function fit() {
                    s = fitS;
                    tx = (cw - img.naturalWidth * s) / 2;
                    ty = (ch - img.naturalHeight * s) / 2;
                    draw();
                }
                function zoomAt(cx, cy, factor) {
                    var ns = Math.min(MAX_ZOOM, Math.max(fitS * 0.5, s * factor));
                    tx = cx - (cx - tx) * ns / s;
                    ty = cy - (cy - ty) * ns / s;
                    s = ns;
                    draw();
                }
                function toScreen(p) { return [p[0] * s + tx, p[1] * s + ty]; }
                function local(e) {
                    var r = canvas.getBoundingClientRect();
                    return [e.clientX - r.left, e.clientY - r.top];
                }
                function hitPoint(xy) {
                    for (var i = pts.length - 1; i >= 0; i--) {
                        var q = toScreen(pts[i]);
                        if (Math.hypot(q[0] - xy[0], q[1] - xy[1]) <= HIT_PX) return i;
                    }
                    return -1;
                }
                function draw() {
                    var dpr = window.devicePixelRatio || 1;
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    ctx.clearRect(0, 0, cw, ch);
                    // Show real pixels (no smoothing) once zoomed past 2x.
                    ctx.imageSmoothingEnabled = s < 2;
                    ctx.drawImage(img, tx, ty, img.naturalWidth * s, img.naturalHeight * s);
                    var sp = pts.map(toScreen);
                    if (sp.length === 2) {
                        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.6)';
                        ctx.beginPath(); ctx.moveTo(sp[0][0], sp[0][1]); ctx.lineTo(sp[1][0], sp[1][1]); ctx.stroke();
                        ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffd60a';
                        ctx.beginPath(); ctx.moveTo(sp[0][0], sp[0][1]); ctx.lineTo(sp[1][0], sp[1][1]); ctx.stroke();
                    }
                    sp.forEach(function(q, i) {
                        // Ring + crosshair with a gap, so the marked spot stays visible.
                        var arms = [[-22, -5], [5, 22]];
                        [['rgba(255,255,255,0.9)', 3], [COLORS[i], 1.5]].forEach(function(st) {
                            ctx.strokeStyle = st[0]; ctx.lineWidth = st[1];
                            ctx.beginPath(); ctx.arc(q[0], q[1], 9, 0, 2 * Math.PI); ctx.stroke();
                            arms.forEach(function(a) {
                                ctx.beginPath(); ctx.moveTo(q[0] + a[0], q[1]); ctx.lineTo(q[0] + a[1], q[1]); ctx.stroke();
                                ctx.beginPath(); ctx.moveTo(q[0], q[1] + a[0]); ctx.lineTo(q[0], q[1] + a[1]); ctx.stroke();
                            });
                        });
                        ctx.font = 'bold 13px sans-serif';
                        ctx.lineWidth = 3; ctx.strokeStyle = 'white'; ctx.strokeText(String(i + 1), q[0] + 11, q[1] - 11);
                        ctx.fillStyle = COLORS[i]; ctx.fillText(String(i + 1), q[0] + 11, q[1] - 11);
                    });
                }

                canvas.addEventListener('wheel', function(e) {
                    e.preventDefault();
                    var xy = local(e);
                    // ctrlKey = trackpad pinch, which sends small deltas.
                    zoomAt(xy[0], xy[1], Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
                }, { passive: false });
                canvas.addEventListener('pointerdown', function(e) {
                    if (e.button !== 0) return;
                    var xy = local(e);
                    var hit = hitPoint(xy);
                    canvas.setPointerCapture(e.pointerId);
                    drag = { idx: hit, x0: xy[0], y0: xy[1], tx0: tx, ty0: ty, moved: false };
                });
                canvas.addEventListener('pointermove', function(e) {
                    var xy = local(e);
                    if (!drag) {
                        canvas.style.cursor = hitPoint(xy) >= 0 ? 'move' : 'crosshair';
                        return;
                    }
                    var dx = xy[0] - drag.x0, dy = xy[1] - drag.y0;
                    if (!drag.moved && Math.hypot(dx, dy) < CLICK_SLOP) return;
                    drag.moved = true;
                    if (drag.idx >= 0) {
                        pts[drag.idx] = [(xy[0] - tx) / s, (xy[1] - ty) / s];
                    } else {
                        canvas.style.cursor = 'grabbing';
                        tx = drag.tx0 + dx; ty = drag.ty0 + dy;
                    }
                    draw();
                });
                canvas.addEventListener('pointerup', function(e) {
                    if (!drag) return;
                    var xy = local(e);
                    if (drag.idx >= 0 && drag.moved) {
                        report();
                    } else if (drag.idx < 0 && !drag.moved) {
                        // Click: 1st → point 1, 2nd → point 2, 3rd starts over.
                        var p = [(xy[0] - tx) / s, (xy[1] - ty) / s];
                        pts = pts.length >= 2 ? [p] : pts.concat([p]);
                        report(); draw();
                    }
                    drag = null;
                    canvas.style.cursor = 'crosshair';
                });
                root.querySelectorAll('.pm-calib-tools button').forEach(function(btn) {
                    btn.addEventListener('click', function() {
                        var act = btn.dataset.act;
                        if (act === 'fit') fit();
                        else zoomAt(cw / 2, ch / 2, act === 'in' ? 1.6 : 1 / 1.6);
                    });
                });
                new ResizeObserver(resize).observe(root);
                img.onload = resize;
                img.src = root.dataset.src;
                report();   // new image → no points yet
            }
            new MutationObserver(initCalibViewer).observe(document.body, { childList: true, subtree: true });

            // ── Sleep / reconnect recovery banner ──────────────────────────────
            // When the laptop wakes from sleep (lid opens, screen-on, etc.) the
            // browser fires visibilitychange: hidden → visible.  If a pipeline
            // was running, Gradio's SSE stream is orphaned and the UI freezes.
            // Show a banner so the user knows what happened and how to recover.
            // Track timestamp so we can ignore normal tab switches (< 30 s hidden).
            var _hiddenAt = 0;
            var _MIN_SLEEP_MS = 30000;

            function _isPipelineRunning() {
                // The Stop button is visible and enabled only while a run is active.
                var buttons = document.querySelectorAll('button');
                for (var i = 0; i < buttons.length; i++) {
                    var btn = buttons[i];
                    if (btn.offsetParent !== null &&
                        !btn.disabled &&
                        btn.textContent.trim().startsWith('Stop')) {
                        return true;
                    }
                }
                return false;
            }

            function _showReconnectBanner() {
                if (document.getElementById('mothbot-reconnect-banner')) return;
                var banner = document.createElement('div');
                banner.id = 'mothbot-reconnect-banner';
                banner.style.cssText = [
                    'position:fixed', 'top:64px', 'left:50%',
                    'transform:translateX(-50%)',
                    'background:#e65100', 'color:#fff',
                    'padding:12px 18px', 'border-radius:8px',
                    'z-index:99999', 'box-shadow:0 4px 16px rgba(0,0,0,.4)',
                    'font:14px/1.5 sans-serif', 'max-width:520px',
                    'text-align:center'
                ].join(';');
                var msg = document.createElement('span');
                msg.innerHTML = '<strong>⚠️ Connection restored after sleep</strong> — '
                    + 'The pipeline is still running in the background. '
                    + 'If the output is frozen, click <strong>Stop</strong> then re-run the current stage.';
                var closeBtn = document.createElement('button');
                closeBtn.textContent = '✕';
                closeBtn.style.cssText = 'background:#fff;color:#e65100;border:none;'
                    + 'border-radius:4px;padding:3px 9px;cursor:pointer;'
                    + 'font-weight:bold;margin-left:12px';
                closeBtn.onclick = function() {
                    var el = document.getElementById('mothbot-reconnect-banner');
                    if (el) el.remove();
                };
                banner.appendChild(msg);
                banner.appendChild(closeBtn);
                document.body.appendChild(banner);
                // Auto-dismiss after 3 s
                setTimeout(function() {
                    var el = document.getElementById('mothbot-reconnect-banner');
                    if (el) el.remove();
                }, 3000);
            }

            document.addEventListener('visibilitychange', function() {
                if (document.visibilityState === 'hidden') {
                    _hiddenAt = Date.now();
                } else if (_hiddenAt > 0) {
                    var hiddenMs = Date.now() - _hiddenAt;
                    _hiddenAt = 0;
                    // Only react to genuine sleep/suspend (hidden > 30 s).
                    // Normal tab switches are milliseconds and should be ignored.
                    if (hiddenMs >= _MIN_SLEEP_MS) {
                        setTimeout(function() {
                            if (_isPipelineRunning()) _showReconnectBanner();
                        }, 2000);
                    }
                }
            });
        }
        """,
        css="""
            /* Collection labels (formatted by the js above) */
            #collection-choices .mb-source-label { display: none; }
            #collection-choices .mb-formatted-label { white-space: pre-wrap; }
            #collection-choices .mb-earlier-run { font-size: 0.85em; opacity: 0.75; }
            /* Pixel Mass calibration viewer (driven by the js above) */
            #pm-calib-points { display: none !important; }
            #pm-calib-viewer { position: relative; width: 100%; background: #111; border-radius: 8px; overflow: hidden; }
            #pm-calib-viewer canvas { display: block; width: 100%; touch-action: none; cursor: crosshair; }
            #pm-calib-viewer .pm-calib-tools { position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; }
            #pm-calib-viewer .pm-calib-tools button {
                min-width: 32px; height: 28px; padding: 0 8px; border: none; border-radius: 6px;
                background: rgba(0,0,0,0.6); color: #fff; font: 600 14px sans-serif; cursor: pointer;
            }
            #pm-calib-viewer .pm-calib-tools button:hover { background: rgba(0,0,0,0.8); }
            #pm-calib-viewer .pm-calib-hint {
                position: absolute; left: 8px; bottom: 8px; padding: 3px 8px; border-radius: 6px;
                background: rgba(0,0,0,0.55); color: #fff; font: 12px sans-serif; pointer-events: none;
            }
            .pm-calib-empty { padding: 48px 16px; text-align: center; opacity: 0.7; border: 1px dashed #888; border-radius: 8px; }
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
            # Where the automatic "Process" run is (or where it stopped).
            auto_run_status = gr.Markdown(value="", visible=False)
            gr.HTML("<div style='flex:1'></div>")  # spacer
            quit_btn = gr.Button("Quit Mothbot", variant="stop", size="sm", scale=0, min_width=160)
            quit_confirm_row = gr.Row(visible=False)
            with quit_confirm_row:
                quit_yes_btn = gr.Button("Yes, quit", variant="stop",     size="sm", scale=0, min_width=120)
                quit_no_btn  = gr.Button("Cancel",    variant="secondary", size="sm", scale=0, min_width=100)

        with gr.Tabs(selected="setup", elem_id="mothbot-tabs") as main_tabs:
            # ~~~~~~~~~~~~ Setup TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Setup", id="setup"):
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
                        with gr.Row():
                            deployment_browse_btn = gr.Button(
                                "Pick a Datasets Folder", size="sm", variant="primary", scale=3,
                            )
                            refresh_btn = gr.Button(
                                "↻ Refresh", size="sm", variant="secondary", scale=1, min_width=100,
                            )
                        classify_link = gr.HTML(value="", visible=False)
                        with gr.Row():
                            classify_open_btn = gr.Button(
                                "🦋 Open in Mothbot Classify", size="sm", variant="secondary", visible=False,
                            )
                        classify_open_status = gr.Markdown(value="", visible=False)
                        with gr.Group():
                            status = gr.Textbox(
                                label="Error", lines=3, interactive=False, visible=False
                            )
                            folder_choices = gr.CheckboxGroup(
                                label="Image Collections Found (select which to process)",
                                elem_id="collection-choices",
                                choices=[],
                                value=[],
                                interactive=True,
                                visible=False,
                            )
                            toggle_all_btn = gr.Button(
                                "Select All", size="sm", visible=False
                            )
                        continue_process_btn = gr.Button(
                            "▶ Process all steps automatically",
                            variant="primary",
                            interactive=False,
                            visible=False,
                        )

                    with gr.Column():
                        gr.Markdown("### Additional Processing Files:")
                        with gr.Row():
                            yolo_model_path = gr.Dropdown(
                                choices=BUNDLED_MODEL_CHOICES,
                                value=DEFAULT_YOLO_MODEL,
                                label="Detection Model",
                                allow_custom_value=True,
                                info="Select a bundled model or browse / paste a custom .pt path.",
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


                # NOTE: deployment_browse_btn.click and deployment_path.change are
                # wired after all tabs are defined (see below) because they reference
                # legacy_converter_tab which is defined later in the tab list.
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
            # ~~~~~~~~~~~~ DETECTION TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Detect", id="detect") as detect_tab:
                with gr.Row():
                    det_model_path_mirror = gr.Dropdown(
                        choices=BUNDLED_MODEL_CHOICES,
                        value=DEFAULT_YOLO_MODEL,
                        label="Detection Model",
                        allow_custom_value=True,
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
                    DELETE_OLD_MODEL_PATCHES = gr.Checkbox(
                        value=False,
                        label="Delete old detection patches if using new model",
                    )
                DET_run_btn = gr.Button("Run Detection", variant="primary")
                with gr.Row():
                    DET_output_box = gr.Textbox(label="Detection Output", lines=20, scale=2)
                    DET_preview_img = gr.Image(
                        label="Live Detection Preview",
                        visible=False,
                        scale=1,
                        show_download_button=False,
                    )

                continue_cluster_btn = gr.Button(
                    "Continue to Cluster", variant="primary", interactive=False
                )

                _det_inputs = [
                    selected_paths,
                    yolo_model_path,
                    imgsz,
                    OVERWRITE_PREV_BOT_DETECTIONS,
                    DELETE_OLD_MODEL_PATCHES,
                    external_keys_state,
                ]
                _det_outputs = [DET_output_box, continue_cluster_btn, stop_btn, DET_preview_img]
                DET_run_btn.click(
                    fn=_manual_run(run_detection_with_continue),
                    inputs=_det_inputs,
                    outputs=_det_outputs,
                )

                continue_cluster_btn.click(
                    fn=go_to_cluster_tab,
                    inputs=[],
                    outputs=[main_tabs],
                )

            # ~~~~~~~~~~~~ Cluster Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Cluster Perceptually", id="cluster") as cluster_tab:
                cluster_run_btn = gr.Button("Cluster Perceptually", variant="primary")
                cluster_output_box = gr.Textbox(label="Cluster Output", lines=20)
                continue_id_btn = gr.Button(
                    "Continue to ID", variant="primary", interactive=False
                )
                _cluster_inputs = [selected_paths]
                _cluster_outputs = [cluster_output_box, continue_id_btn, stop_btn]
                cluster_run_btn.click(
                    fn=_manual_run(run_cluster_with_continue),
                    inputs=_cluster_inputs,
                    outputs=_cluster_outputs,
                )

                continue_id_btn.click(
                    fn=go_to_id_tab,
                    inputs=[],
                    outputs=[main_tabs],
                )

            # ~~~~~~~~~~~~ IDENTIFICATION TAB ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("ID", id="id") as id_tab:
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
                        value=DEFAULT_SPECIES_CSV,
                        interactive=True,
                    )
                    id_species_browse_mirror = gr.Button("Browse", size="sm", scale=0, min_width=100)
                with gr.Row():
                    with gr.Column(scale=3):
                        blur_threshold = gr.Slider(
                            minimum=0, maximum=100, value=100, step=1,
                            label="Blurriness threshold (0 = sharp, 100 = blurriest)",
                            info="Blurriness is whichever is worse: how little fine detail the insect has "
                                 "for its size (out of focus or too small), or how much its edges all run one "
                                 "way (motion streaks). Only patches at or below the threshold are identified; "
                                 "100 identifies everything. Around 70-80 skips most too-blurry patches and "
                                 "almost no sharp ones.",
                        )
                        blur_example_caption = gr.Markdown("")
                    blur_example_img = gr.Image(
                        label="Example patch near this blurriness",
                        interactive=False, show_download_button=False,
                        height=180, scale=1,
                    )
                blur_examples_state = gr.State([])
                ID_run_btn = gr.Button("Run Identification", variant="primary")
                ID_output_box = gr.Textbox(label="Identification Output", lines=20)

                _id_inputs = [
                    selected_paths,
                    id_species_mirror,
                    taxa_output,
                    ID_HUMANDETECTIONS,
                    ID_BOTDETECTIONS,
                    OVERWRITE_PREV_BOT_IDENTIFICATIONS,
                    blur_threshold,
                ]
                _id_outputs = [ID_output_box, stop_btn]
                ID_run_btn.click(
                    fn=_manual_run(run_ID),
                    inputs=_id_inputs,
                    outputs=_id_outputs,
                )
                # Sample example patches from the chosen collections when the tab
                # opens, then show the one nearest the threshold as it moves.
                id_tab.select(
                    fn=load_blur_examples,
                    inputs=[selected_paths, blur_threshold],
                    outputs=[blur_examples_state, blur_example_img, blur_example_caption],
                )
                blur_threshold.change(
                    fn=show_blur_example,
                    inputs=[blur_examples_state, blur_threshold],
                    outputs=[blur_example_img, blur_example_caption],
                )

            # ~~~~~~~~~~~~ Metadata Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Insert Metadata", id="metadata") as metadata_tab:
                metadata_mode = gr.Radio(
                    choices=["CSV File", "Manual Entry"],
                    value="CSV File",
                    label="Metadata Source",
                )

                with gr.Group(visible=True) as meta_csv_group:
                    with gr.Row():
                        meta_csv_mirror = gr.Text(
                            label="Metadata field sheet:",
                            interactive=True,
                        )
                        meta_browse_mirror = gr.Button("Browse", size="sm", scale=0, min_width=100)

                with gr.Group(visible=False) as meta_manual_group:
                    meta_manual_folder_choices = gr.CheckboxGroup(
                        label="Apply to which nights (select the folders this metadata applies to):",
                        choices=[],
                        value=[],
                        interactive=True,
                    )
                    gr.Markdown("### Enter deployment metadata manually")
                    with gr.Row():
                        meta_project = gr.Text(label="Project", interactive=True)
                        meta_site = gr.Text(label="Site", interactive=True)
                        meta_device = gr.Text(label="Device (Mothbox name)", interactive=True)
                    with gr.Row():
                        meta_deployment_date = gr.DateTime(
                            label="Deployment Date",
                            include_time=False,
                            type="string",
                        )
                        meta_collect_date = gr.DateTime(
                            label="Collect Date",
                            include_time=False,
                            type="string",
                        )
                        meta_height = gr.Text(label="Height Above Ground", interactive=True)
                    meta_deployment_name = gr.Text(
                        label="Deployment Name (auto-generated from Project + Site + Device + Date)",
                        interactive=False,
                    )
                    with gr.Row():
                        meta_latitude = gr.Text(label="Latitude", interactive=True)
                        meta_longitude = gr.Text(label="Longitude", interactive=True)
                        meta_crew = gr.Text(label="Crew", interactive=True)
                    with gr.Row():
                        meta_habitat = gr.Text(label="Habitat", interactive=True)
                        meta_attractor = gr.Text(label="Attractor", interactive=True)
                        meta_attractor_location = gr.Text(label="Attractor Location", interactive=True)
                    with gr.Row():
                        meta_firmware = gr.Text(label="Firmware", interactive=True)
                        meta_utc = gr.Text(label="UTC Offset", interactive=True)
                        meta_schedule = gr.Text(label="Schedule", interactive=True)
                    with gr.Row():
                        meta_storage_loc = gr.Text(label="Data Storage Location", interactive=True)
                    meta_notes = gr.Textbox(label="Notes", interactive=True, lines=3)

                overwrite_metadata = gr.Checkbox(
                    value=True,
                    label="Overwrite existing metadata (uncheck to skip detections that already have metadata)",
                )

                metadata_run_btn = gr.Button("Insert Metadata", variant="primary")
                metadata_output_box = gr.Textbox(
                    label="Insert Metadata Output", lines=20
                )

                metadata_mode.change(
                    fn=lambda mode: (
                        gr.update(visible=(mode == "CSV File")),
                        gr.update(visible=(mode == "Manual Entry")),
                    ),
                    inputs=[metadata_mode],
                    outputs=[meta_csv_group, meta_manual_group],
                )

                for _dep_trigger in [meta_project, meta_site, meta_device, meta_deployment_date]:
                    _dep_trigger.change(
                        fn=generate_deployment_name,
                        inputs=[meta_project, meta_site, meta_device, meta_deployment_date],
                        outputs=[meta_deployment_name],
                    )

                meta_manual_folder_choices.change(
                    fn=load_metadata_for_preview,
                    inputs=[meta_manual_folder_choices, mapping_state, dataset_root_state],
                    outputs=[
                        meta_deployment_name, meta_latitude, meta_longitude,
                        meta_crew, meta_project, meta_site,
                        meta_habitat, meta_device, meta_firmware,
                        meta_utc, meta_deployment_date, meta_collect_date,
                        meta_attractor, meta_attractor_location, meta_height,
                        meta_schedule, meta_storage_loc, meta_notes,
                    ],
                )

                meta_latitude.change(
                    fn=maybe_split_latlong,
                    inputs=[meta_latitude],
                    outputs=[meta_latitude, meta_longitude],
                )
                meta_longitude.change(
                    fn=maybe_split_latlong,
                    inputs=[meta_longitude],
                    outputs=[meta_latitude, meta_longitude],
                )

                _metadata_inputs = [
                        selected_paths,
                        metadata_mode,
                        meta_csv_mirror,
                        overwrite_metadata,
                        meta_manual_folder_choices,
                        mapping_state,
                        external_keys_state,
                        dataset_root_state,
                        meta_deployment_name,
                        meta_latitude,
                        meta_longitude,
                        meta_crew,
                        meta_project,
                        meta_site,
                        meta_habitat,
                        meta_device,
                        meta_firmware,
                        meta_utc,
                        meta_deployment_date,
                        meta_collect_date,
                        meta_attractor,
                        meta_attractor_location,
                        meta_height,
                        meta_schedule,
                        meta_storage_loc,
                        meta_notes,
                ]
                _metadata_outputs = [metadata_output_box, stop_btn]
                metadata_run_btn.click(
                    fn=_manual_run(run_metadata),
                    inputs=_metadata_inputs,
                    outputs=_metadata_outputs,
                )

            # ~~~~~~~~~~~~ Exif Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Insert Exif", id="exif") as exif_tab:
                exif_run_btn = gr.Button("Insert Exif (Optional)", variant="primary")
                exif_output_box = gr.Textbox(label="Insert Exif Output", lines=20)

                _exif_inputs = [selected_paths]
                _exif_outputs = [exif_output_box, stop_btn]
                exif_run_btn.click(
                    fn=_manual_run(run_exif),
                    inputs=_exif_inputs,
                    outputs=_exif_outputs,
                )

            # ~~~~~~~~~~~~ Pixel Mass Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Pixel Mass", id="pixel_mass") as pixel_mass_tab:
                pm_point1_state     = gr.State(None)   # [x, y] of point 1 (original-image pixels)
                pm_point2_state     = gr.State(None)   # [x, y] of point 2 (original-image pixels)

                with gr.Accordion("Step 1: Set Scale Calibration", open=True) as pm_step1_accordion:
                    gr.Markdown(
                        "Zoom in on a ruler or known object in the source image below and click two points "
                        "(drag a point to fine-tune it). "
                        "Enter the real-world distance and click **Apply Calibration**. "
                        "Or type a known **pixels per mm** value directly and apply."
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            pm_calib_viewer = gr.HTML(_calib_viewer_html())
                            # Written by the viewer's JS; hidden with CSS (a
                            # visible=False component isn't in the page at all).
                            pm_calib_points = gr.Textbox(value="[]", elem_id="pm-calib-points")
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
                    pm_only_identified = gr.Checkbox(
                        label="Only measure identified patches (skip ones ID left unidentified, e.g. too blurry)",
                        value=False,
                    )
                pm_model_dropdown = gr.Dropdown(
                    label="Background removal model",
                    choices=[
                        ("birefnet-general — best quality, slowest", "birefnet-general"),
                        ("birefnet-general-lite — good quality, faster", "birefnet-general-lite"),
                        ("isnet-general-use — medium quality, faster", "isnet-general-use"),
                        ("u2netp — lowest quality, fastest", "u2netp"),
                    ],
                    value="birefnet-general-lite",
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
                    pm_calib_viewer, pm_calib_status,
                    pm_point1_state, pm_point2_state,
                    pm_point1_label, pm_point2_label, pm_pixel_dist_label,
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

                pm_calib_points.change(
                    fn=calibration_points_changed,
                    inputs=[pm_calib_points],
                    outputs=[pm_point1_state, pm_point2_state,
                             pm_point1_label, pm_point2_label, pm_pixel_dist_label],
                )

                pm_calibrate_btn.click(
                    fn=apply_calibration,
                    inputs=[selected_paths, pm_point1_state, pm_point2_state,
                            pm_real_dist, pm_pixels_per_mm],
                    outputs=[pm_pixels_per_mm, pm_calib_status],
                )

                pm_run_btn.click(
                    fn=_manual_run(run_pixel_mass_ui),
                    inputs=[selected_paths, pm_pixels_per_mm, pm_overwrite_nobg, pm_overwrite_pixmass, pm_model_dropdown, pm_only_identified],
                    outputs=[pm_output_box, stop_btn, pm_step1_accordion, pm_preview_img],
                )

            # ~~~~~~~~~~~~ Legacy Converter Tab ~~~~~~~~~~~~~~~~~~~~~~
            with gr.Tab("Legacy Converter", id="legacy_converter", visible=False) as legacy_converter_tab:
                gr.Markdown(
                    "**Convert legacy-format datasets** to the current `_processed/` layout.\n\n"
                    "Old Mothbot versions stored patches in a `patches/` subfolder and wrote "
                    "`_botdetection.json` files next to source images. "
                    "This tool moves those outputs into the `_processed/` mirror tree so the "
                    "dataset works with current Mothbot Process and Mothbot Classify.\n\n"
                    "Legacy collections are detected automatically when you scan a folder in the "
                    "**Setup** tab. Use **↻ Refresh** there to re-scan after making changes."
                )
                lc_folder_choices = gr.CheckboxGroup(
                    label="Legacy collections found (select which to convert)",
                    choices=[],
                    value=[],
                    visible=False,
                )
                lc_delete_originals = gr.Checkbox(
                    label="Delete original files after converting (⚠️ irreversible — back up first)",
                    value=False,
                )
                with gr.Row():
                    lc_run_btn = gr.Button("Convert Selected", variant="primary", interactive=False)
                lc_output_box = gr.Textbox(label="Conversion Output", lines=20, interactive=False)

                lc_folder_choices.change(
                    fn=lambda v: gr.update(interactive=bool(v)),
                    inputs=[lc_folder_choices],
                    outputs=[lc_run_btn],
                )
                lc_run_btn.click(
                    fn=run_legacy_converter_ui,
                    inputs=[lc_folder_choices, dataset_root_state, lc_delete_originals],
                    outputs=[lc_output_box, stop_btn],
                )

            # ── "Process all steps automatically" ──────────────────────────────
            # Drives the real stage tabs in order: switch to the tab, then run that
            # tab's own handler with its own inputs and outputs — exactly what the
            # user would get clicking through by hand. Each tab's log, preview and
            # Run button stay live, and the run stops on the tab where it stopped
            # (Stop, an error, or a missing metadata source).
            _AUTO_STEPS = [
                ("detect", "Detect", run_detection_with_continue, _det_inputs, _det_outputs),
                ("cluster", "Cluster", run_cluster_with_continue, _cluster_inputs, _cluster_outputs),
                ("id", "ID", run_ID, _id_inputs, _id_outputs),
                ("metadata", "Insert Metadata", _auto_metadata_step, _metadata_inputs, _metadata_outputs),
                ("exif", "Insert Exif", run_exif, _exif_inputs, _exif_outputs),
            ]
            _chain = continue_process_btn.click(
                fn=_begin_auto_run,
                inputs=[selected_paths],
                outputs=[auto_run_status],
            )
            for _n, (_tab_id, _label, _handler, _inputs, _outputs) in enumerate(_AUTO_STEPS, start=1):
                _chain = _chain.then(
                    fn=_auto_goto(_tab_id, _label, _n, len(_AUTO_STEPS)),
                    inputs=[],
                    outputs=[main_tabs, auto_run_status],
                ).then(
                    fn=_auto_step(_handler, len(_outputs)),
                    inputs=_inputs,
                    outputs=_outputs,
                )
            _chain.then(fn=_end_auto_run, inputs=[], outputs=[auto_run_status, stop_btn])

        # ── Stop button ────────────────────────────────────────────────────────
        def do_cancel():
            request_cancel()
            # run_in_thread clears its cancel flag once honored; this one persists
            # so the remaining folders and automatic-run steps are skipped too.
            _RUN_STATE["stop"] = True
            return gr.update(value="⛔ Stopping…", interactive=False)

        stop_btn.click(fn=do_cancel, inputs=[], outputs=[stop_btn])

        # ── Deployment folder scan (wired here so legacy_converter_tab is in scope) ──
        _scan_outputs = [
            status,
            folder_choices,
            mapping_state,
            toggle_label_state,
            continue_process_btn,
            selected_paths,
            toggle_all_btn,
            external_keys_state,
            dataset_root_state,
            legacy_converter_tab,
            lc_folder_choices,
            lc_run_btn,
            meta_manual_folder_choices,
        ]
        deployment_browse_btn.click(
            fn=browse_deployment_folder,
            inputs=[deployment_path],
            outputs=[deployment_path, picker_error_state],
        ).then(
            fn=scan_deployment_folder,
            inputs=[deployment_path, picker_error_state],
            outputs=_scan_outputs,
        )
        deployment_path.change(
            fn=scan_deployment_folder_on_change,
            inputs=[deployment_path],
            outputs=_scan_outputs,
        )
        # Show the "open in Classify" handoff link whenever the dataset folder changes.
        def _classify_handoff_ui(folder_path):
            link = classify_handoff_html(folder_path)
            return link, gr.update(visible=bool(link.get("visible"))), gr.update(value="", visible=False)

        deployment_path.change(
            fn=_classify_handoff_ui,
            inputs=[deployment_path],
            outputs=[classify_link, classify_open_btn, classify_open_status],
        )
        # Opens the link in a Chromium browser on this machine (Classify's folder
        # picker needs the File System Access API, which Firefox/Safari lack).
        classify_open_btn.click(
            fn=open_classify_in_chrome,
            inputs=[deployment_path],
            outputs=[classify_open_status],
        )
        refresh_btn.click(
            fn=scan_deployment_folder_on_change,
            inputs=[deployment_path],
            outputs=_scan_outputs,
        )

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

        # Setup folder selection → manual-entry night picker (keeps selection in sync)
        folder_choices.change(
            fn=lambda v: gr.update(value=v),
            inputs=[folder_choices],
            outputs=[meta_manual_folder_choices],
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


def generate_deployment_name(project, site, device, deploy_date):
    """Build deployment name as project_site_device_YYYY-MM-DD (matching the CSV formula)."""
    from datetime import datetime as _dt
    project = (project or "").strip()
    site = (site or "").strip()
    device = (device or "").strip()
    date_str = (deploy_date or "").strip()

    if date_str:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                date_str = _dt.strptime(date_str, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    parts = [p for p in [project, site, device, date_str] if p]
    return "_".join(parts)


def maybe_split_latlong(value):
    """If value looks like 'lat, lon' (two numbers separated by a comma),
    split and return (lat, lon) for the two text boxes.  Otherwise no-op."""
    import re as _re
    value = (value or "").strip()
    m = _re.match(r"^([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)$", value)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return gr.update(), gr.update()


def load_metadata_for_preview(selected_keys, mapping, dataset_root):
    """Read existing metadata from the first selected night and pre-populate the manual-entry form."""
    import json as _json
    from core.common import find_detection_matches_processed

    EMPTY = tuple(gr.update() for _ in range(18))

    if not selected_keys or not mapping:
        return EMPTY

    folder_path = mapping.get(selected_keys[0])
    if not folder_path or not os.path.isdir(folder_path):
        return EMPTY

    dr = dataset_root or folder_path
    try:
        hu_pairs, bot_pairs = find_detection_matches_processed(dr, source_folder=folder_path)
    except Exception:
        return EMPTY

    for _, json_path in (hu_pairs + bot_pairs):
        try:
            with open(json_path) as f:
                data = _json.load(f)
            if not data.get("field_sheet_metadata"):
                continue
            project = str(data.get("project", "") or "")
            site = str(data.get("site", "") or "")
            device = str(data.get("device", "") or "")
            deploy_date = str(data.get("deployment_date", "") or "")
            auto_name = generate_deployment_name(project, site, device, deploy_date)
            return (
                gr.update(value=auto_name),
                gr.update(value=str(data.get("latitude", "") or "")),
                gr.update(value=str(data.get("longitude", "") or "")),
                gr.update(value=str(data.get("crew", "") or "")),
                gr.update(value=project),
                gr.update(value=site),
                gr.update(value=str(data.get("habitat", "") or "")),
                gr.update(value=device),
                gr.update(value=str(data.get("firmware", "") or "")),
                gr.update(value=str(data.get("UTC", "") or "")),
                gr.update(value=deploy_date),
                gr.update(value=str(data.get("collect_date", "") or "")),
                gr.update(value=str(data.get("attractor", "") or "")),
                gr.update(value=str(data.get("attractor_location", "") or "")),
                gr.update(value=str(data.get("ground_height", "") or "")),
                gr.update(value=str(data.get("schedule", "") or "")),
                gr.update(value=str(data.get("data_storage_location", "") or "")),
                gr.update(value=str(data.get("notes", "") or "")),
            )
        except Exception:
            continue

    return EMPTY


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
    # ONNX support is temporarily disabled — our direct ONNX Runtime inference
    # path produces correct detection counts but the OBB coordinate inverse-
    # transform (undoing letterbox + rotation) has an off-by-one that makes
    # patches land in the wrong place or outside the image entirely, causing
    # black crops and warpAffine assertion errors.  .pt models work correctly.
    # Re-enable by adding ("ONNX model", "*.onnx") back to filetypes once fixed.
    return _browse_file(
        current_path, filetypes=[("YOLO model", "*.pt"), ("PyTorch model", "*.pt"), ("All files", "*.*")]
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


_PATCH_NAME = re.compile(r"_\d+_(.+?)\.jpe?g$", re.IGNORECASE)
_ARCHIVED_RUN_JSON = re.compile(r"_botdetection_(.+)\.json$")


def _run_slug(model_name):
    """Detector name as used in archived-run file names (see detect._model_archive_path)."""
    return model_name.removesuffix(".pt").replace(" ", "_")


def _short_model(model_name):
    return model_name.removesuffix(".pt").removeprefix("Mothbot_") or "unknown model"


def _sample_run_status(folder, json_names, max_reads=40):
    """(model, stage flags) from the first of *json_names* that has detections."""
    flags = {"clustered": False, "identified": False, "pixel_mass": False}
    model = None
    for name in json_names[:max_reads]:
        try:
            with open(os.path.join(folder, name)) as fh:
                data = json.load(fh)
        except Exception:
            continue
        model = model or data.get("version")
        shapes = data.get("shapes") or []
        if shapes:
            flags["clustered"] = any(sh.get("clusterID") is not None for sh in shapes)
            flags["identified"] = any(sh.get("identifier_bot", "") not in ("", None) for sh in shapes)
            flags["pixel_mass"] = any("pixel_mass_pixels" in sh for sh in shapes)
            break
    return model, flags


def _detection_run_summaries(processed_mirror):
    """Per detection run in a collection's processed folder: model, photos covered,
    patches, and which later stages ran on it. Re-running Detect with another model
    archives the previous run beside the current one, so counting every JSON and patch
    together would add the runs up. Counts come from file names; stage flags from one
    sampled JSON per run, so scanning stays fast.
    """
    summary = {"current": None, "archived": [], "human_patches": 0, "human_photos": 0}
    if not os.path.isdir(processed_mirror):
        return summary
    try:
        names = sorted(os.listdir(processed_mirror))
    except OSError:
        return summary

    current_jsons = [n for n in names if n.endswith("_botdetection.json")]
    archived_jsons = {}
    for n in names:
        m = _ARCHIVED_RUN_JSON.search(n)
        if m:
            archived_jsons.setdefault(m.group(1), []).append(n)
    patches_by_slug = {}
    human_photos = set()
    for n in names:
        if n.endswith("_HumanDetection.jpg"):
            summary["human_patches"] += 1
            human_photos.add(n.rsplit("_", 2)[0])  # <photo>_<i>_HumanDetection.jpg
            continue
        m = _PATCH_NAME.search(n)
        if m:
            slug = _run_slug(m.group(1))
            patches_by_slug[slug] = patches_by_slug.get(slug, 0) + 1
    summary["human_photos"] = len(human_photos)

    if current_jsons:
        model, flags = _sample_run_status(processed_mirror, current_jsons)
        slug = _run_slug(model) if model else None
        summary["current"] = {"model": model or "unknown model", "photos": len(current_jsons),
                              "patches": patches_by_slug.get(slug, 0) if slug else 0, **flags}
    for slug, jsons in sorted(archived_jsons.items()):
        model, flags = _sample_run_status(processed_mirror, jsons)
        summary["archived"].append({"model": model or slug, "photos": len(jsons),
                                    "patches": patches_by_slug.get(slug, 0), **flags})
    return summary


def _format_run(run, total_photos, prefix=""):
    stages = "  ".join(tag for flag, tag in ((run["clustered"], "✓ Cluster"), (run["identified"], "✓ ID"),
                                             (run["pixel_mass"], "✓ PixMass")) if flag)
    coverage = f" on {run['photos']}/{total_photos} photos" if total_photos and run["photos"] < total_photos else ""
    text = f"{prefix}{_short_model(run['model'])}: 🦋 {run['patches']}{coverage}"
    return f"{text}  {stages}" if stages else text


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
        gr.update(visible=False),  # legacy_converter_tab
        gr.update(choices=[], value=[], visible=False),  # lc_folder_choices
        gr.update(interactive=False),  # lc_run_btn
        gr.update(choices=[], value=[]),  # meta_manual_folder_choices
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
    lc_choices = []

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

        # For external collections the folder IS the processed mirror.
        processed_mirror = p if is_external else os.path.join(folder_path, "_processed", os.path.relpath(p, folder_path))
        ps = _check_pipeline_status(processed_mirror)
        runs = _detection_run_summaries(processed_mirror)
        total_photos = 0 if is_external else jpeg_count
        parts = [] if is_external else [f"📷 {jpeg_count}"]
        if runs["current"]:
            parts.append(_format_run(runs["current"], total_photos))
        elif not is_external:
            parts.append("not detected yet")
        parts += [_format_run(run, total_photos, prefix="earlier ") for run in runs["archived"]]
        if runs["human_patches"]:
            photos = runs["human_photos"]
            parts.append(f"human: 🦋 {runs['human_patches']} on {photos} photo{'' if photos == 1 else 's'}")
        counts = "  |  ".join(parts)

        # Photo-level stages (the per-run ones are shown with each run above).
        pipeline_tags = "  ".join(tag for flag, tag in ((ps["metadata"], "✓ Meta"), (ps["exif"], "✓ Exif")) if flag)

        if is_external:
            label = f"⚡ {label_prefix}  {counts}  [ext]"
            is_legacy = False
        else:
            is_legacy = Mothbot_LegacyConverter.is_legacy_collection(p)
            legacy_flag = "⚠️ legacy  " if is_legacy else ""
            label = f"{legacy_flag}{label_prefix}  {counts}"
        if pipeline_tags:
            label += f"  |  {pipeline_tags}"

        choices.append((label, value))
        mapping[value] = os.path.abspath(p)
        if is_external:
            external_keys.add(value)
        elif is_legacy:
            patches_dir_lc = os.path.join(p, "patches")
            json_count_lc = len(glob.glob(os.path.join(glob.escape(p), "*_botdetection.json")))
            patch_count_lc = _count_matching_files(patches_dir_lc, ("*.jpg", "*.jpeg")) if os.path.isdir(patches_dir_lc) else 0
            lc_choices.append((
                f"{rel}  ({json_count_lc} JSON, {patch_count_lc} patches)",
                os.path.abspath(p),
            ))

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

    any_legacy = bool(lc_choices)
    status = (
        f"Selected folder: {folder_path}\n"
        f"Found {len(raw_matches)} raw collection(s)"
        + (f" + {len(external_matches)} externally-processed collection(s)." if external_matches else ".")
        + ("\n⚠️  Legacy-format collections detected — see the Legacy Converter tab." if any_legacy else "")
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
        gr.update(visible=any_legacy),  # legacy_converter_tab
        gr.update(choices=lc_choices, value=[c[1] for c in lc_choices], visible=any_legacy),  # lc_folder_choices
        gr.update(interactive=any_legacy),  # lc_run_btn
        gr.update(choices=choices, value=[]),  # meta_manual_folder_choices
    )


CLASSIFY_URL = "https://classify.mothbox.org"

# Classify needs the File System Access API (showDirectoryPicker), which only
# Chromium-based browsers implement. Firefox/Safari can load the page but can't
# open a folder, so we try to launch the link in a Chromium browser directly.
_CHROMIUM_BROWSERS = {
    "darwin": [
        ("Google Chrome", ["open", "-a", "Google Chrome"]),
        ("Microsoft Edge", ["open", "-a", "Microsoft Edge"]),
        ("Brave", ["open", "-a", "Brave Browser"]),
        ("Chromium", ["open", "-a", "Chromium"]),
    ],
    "win32": [
        ("Google Chrome", ["cmd", "/c", "start", "", "chrome"]),
        ("Microsoft Edge", ["cmd", "/c", "start", "", "msedge"]),
    ],
    "linux": [
        ("Google Chrome", ["google-chrome"]),
        ("Chromium", ["chromium"]),
        ("Chromium", ["chromium-browser"]),
        ("Microsoft Edge", ["microsoft-edge"]),
    ],
}


def _classify_handoff_url(folder_path):
    """Return (url, root) for the chosen datasets folder.

    Hand Classify exactly the folder chosen here — not its parent, not a
    sub-folder. It becomes Classify's datasets folder, and Classify lists the
    datasets inside it.
    """
    from urllib.parse import quote

    root = os.path.normpath(folder_path)
    return f"{CLASSIFY_URL}/?root={quote(root)}", root


def open_classify_in_chrome(folder_path):
    """Open the Classify hand-off link, preferring a Chromium browser.

    Process runs locally, so this launches a browser on the user's own machine.
    Falls back to the default browser (with a warning) when no Chromium build is
    found — Classify loads there but its folder picker won't work.
    """
    import shutil
    import subprocess
    import webbrowser

    folder_path = (folder_path or "").strip()
    if not folder_path or not os.path.isdir(folder_path):
        return gr.update(value="Choose a datasets folder first.", visible=True)

    url, _root = _classify_handoff_url(folder_path)

    for name, cmd in _CHROMIUM_BROWSERS.get(sys.platform, []):
        exe = cmd[0]
        if exe in ("open", "cmd"):
            if shutil.which(exe) is None:
                continue
        elif shutil.which(exe) is None:
            continue
        try:
            result = subprocess.run([*cmd, url], capture_output=True, timeout=15)
            if result.returncode == 0:
                return gr.update(value=f"✅ Opened Mothbot Classify in {name}.", visible=True)
        except Exception:
            continue

    webbrowser.open(url)
    return gr.update(
        value=(
            "⚠️ Couldn't find Chrome, Edge, or Brave, so this opened in your default browser. "
            "Classify needs a Chromium-based browser to open folders — if the page can't open your "
            "dataset, paste the link into Chrome."
        ),
        visible=True,
    )


def classify_handoff_html(folder_path):
    """Link that hands the chosen datasets folder off to Mothbot Classify.

    A browser can't open a local folder from a URL, but Classify remembers the
    datasets folder it was pointed at. The folder chosen here is sent as
    ``?root=`` so Classify opens that exact folder — and uses it (display-only)
    to name the folder to pick if it isn't pointed there yet.
    """
    import html

    folder_path = (folder_path or "").strip()
    if not folder_path or not os.path.isdir(folder_path):
        return gr.update(value="", visible=False)

    url, root = _classify_handoff_url(folder_path)
    target = os.path.basename(root) or root
    body = (
        f'<div style="margin:6px 0 2px 0;font-size:14px;line-height:1.5">'
        f'<a href="{html.escape(url)}" target="_blank" rel="noopener" '
        f'style="font-weight:600;text-decoration:none">'
        f'🦋 Open <b>{html.escape(target)}</b> in Mothbot Classify ↗</a>'
        f'<div style="color:#888;font-size:12px;margin-top:2px">'
        f'Use the button for best results — Classify needs Chrome/Edge to open folders. '
        f'Point its datasets folder at <code>{html.escape(root)}</code> if it asks.'
        f'</div></div>'
    )
    return gr.update(value=body, visible=True)


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


# ──────────────────────────────────────────────────────────────
#  Run state: Stop across folders, and the automatic "Process" run
# ──────────────────────────────────────────────────────────────

# stop  — user pressed Stop; skip remaining folders and remaining auto steps.
# error — a stage hit an exception; the automatic run halts on that tab.
# paused — the automatic run needs user input (e.g. a metadata source).
# auto  — an automatic run is in progress.  step — its current step label.
_RUN_STATE = {"stop": False, "error": False, "paused": None, "auto": False, "step": None}


def _stop_requested():
    return _RUN_STATE["stop"]


def _note_stage_error():
    _RUN_STATE["error"] = True


def _manual_run(handler):
    """Wrap a tab's own Run button: a fresh user-started run clears old stop/error state."""
    def wrapped(*args):
        _RUN_STATE.update(stop=False, error=False, paused=None, auto=False, step=None)
        yield from handler(*args)
    return wrapped


def _auto_run_live():
    return (
        _RUN_STATE["auto"]
        and not _RUN_STATE["stop"]
        and not _RUN_STATE["error"]
        and not _RUN_STATE["paused"]
    )


def _auto_step(handler, n_outputs):
    """One step of the automatic run — runs the tab's handler, or does nothing once halted."""
    def wrapped(*args):
        if not _auto_run_live():
            yield tuple(gr.update() for _ in range(n_outputs))
            return
        yield from handler(*args)
    return wrapped


def _auto_goto(tab_id, label, number, total):
    """Switch to the next step's tab — unless the run halted, so the user stays where it stopped."""
    def goto():
        if not _auto_run_live():
            return gr.update(), gr.update()
        _RUN_STATE["step"] = label
        return (
            gr.Tabs(selected=tab_id),
            gr.update(value=f"▶ **Automatic run** — step {number} of {total}: **{label}**", visible=True),
        )
    return goto


def _begin_auto_run(selected_folders):
    if not selected_folders:
        _RUN_STATE.update(auto=False)
        return gr.update(value="Select at least one image collection first.", visible=True)
    _RUN_STATE.update(stop=False, error=False, paused=None, auto=True, step=None)
    return gr.update(value="▶ **Automatic run** starting…", visible=True)


def _end_auto_run():
    step = _RUN_STATE["step"] or "the first step"
    if not _RUN_STATE["auto"]:
        message = gr.update()
    elif _RUN_STATE["stop"]:
        message = gr.update(
            value=f"⛔ **Automatic run stopped** during **{step}** — continue from the {step} tab.",
            visible=True,
        )
    elif _RUN_STATE["error"]:
        message = gr.update(
            value=f"❌ **Automatic run halted** at **{step}** because of an error — see the {step} tab.",
            visible=True,
        )
    elif _RUN_STATE["paused"]:
        message = gr.update(
            value=f"⏸ **Automatic run paused** at **{_RUN_STATE['paused']}** — it needs your input there.",
            visible=True,
        )
    else:
        message = gr.update(value="✅ **Automatic run finished** — all steps completed.", visible=True)
    _RUN_STATE.update(auto=False)
    return message, gr.update(visible=False)


def _auto_metadata_step(selected_folders, metadata_mode, metadata_csv, *rest):
    """Insert Metadata for the automatic run; pauses there if no metadata source was chosen."""
    if metadata_mode == "CSV File" and not (metadata_csv and str(metadata_csv).strip()):
        _RUN_STATE["paused"] = "Insert Metadata"
        yield (
            "⏸ The automatic run paused here: no metadata CSV was chosen.\n"
            "→ Choose a CSV above (or switch to Manual Entry), click Insert Metadata,\n"
            "  then run the Insert Exif tab to finish.\n",
            gr.update(visible=False),
        )
        return
    yield from run_metadata(selected_folders, metadata_mode, metadata_csv, *rest)


def go_to_id_tab():
    return gr.Tabs(selected="id")

def go_to_cluster_tab():
    return gr.Tabs(selected="cluster")

# The calibration viewer (zoom/pan/click, all in the browser — see pmCalibViewer
# in the Blocks js) loads the source image from here. Only a copy of the one
# image being calibrated is ever put in this folder.
_CALIB_VIEW_DIR = Path.home() / ".mothbot" / "calib_view"
_CALIB_VIEW_DIR.mkdir(parents=True, exist_ok=True)
gr.set_static_paths([_CALIB_VIEW_DIR])

_CALIB_HINT = "Scroll or pinch to zoom · drag to pan · click to place points · drag a point to move it"


def _calib_viewer_html(url=None, message="Open this tab with a collection selected to load an image."):
    if not url:
        return f'<div class="pm-calib-empty">{message}</div>'
    return (
        f'<div id="pm-calib-viewer" data-src="{url}">'
        '<canvas></canvas>'
        '<div class="pm-calib-tools"><button data-act="in">+</button>'
        '<button data-act="out">−</button><button data-act="fit">Fit</button></div>'
        f'<div class="pm-calib-hint">{_CALIB_HINT}</div>'
        '</div>'
    )


def _calib_labels(p1, p2):
    """Point / distance readouts. Points are in original-image pixels."""
    import math
    p1_str = f"({p1[0]:.1f}, {p1[1]:.1f})" if p1 else "–"
    p2_str = f"({p2[0]:.1f}, {p2[1]:.1f})" if p2 else "–"
    dist_str = f"{math.dist(p1, p2):.1f} px" if p1 and p2 else "–"
    return p1_str, p2_str, dist_str


def load_image_for_calibration(selected_folders):
    """Put a copy of the first source image where the browser viewer can load it.

    Returns (viewer_html, status, p1, p2, p1_label, p2_label, dist_label).
    """
    import shutil
    import time
    _RESET = (None, None, "–", "–", "–")
    if not selected_folders:
        return (_calib_viewer_html(), "No collection selected.", *_RESET)
    entry = selected_folders[0]
    folder = entry["path"] if isinstance(entry, dict) else entry
    images = find_images_recursive(folder)
    if not images:
        return (_calib_viewer_html(message="No source images found in collection."),
                "No source images found in collection.", *_RESET)
    from PIL import Image as PILImage
    src = images[0]
    with PILImage.open(src) as img:
        w, h = img.size
    ext = Path(src).suffix.lower()
    # Replace the previous copy (only files named calib_source.* live here).
    for old in _CALIB_VIEW_DIR.glob("calib_source.*"):
        old.unlink(missing_ok=True)
    view_path = _CALIB_VIEW_DIR / f"calib_source{ext}"
    shutil.copyfile(src, view_path)
    # Cache-bust: the copy always has the same name.
    url = f"/gradio_api/file={view_path.as_posix()}?v={time.time_ns()}"
    status = f"Loaded: {os.path.basename(src)} ({w}×{h})"
    return (_calib_viewer_html(url), status, *_RESET)


def calibration_points_changed(points_json):
    """The browser viewer reports its points as JSON: [] / [[x, y]] / [[x, y], [x, y]]
    in original-image pixels. Returns (p1, p2, p1_label, p2_label, dist_label)."""
    try:
        pts = [[round(float(v), 1) for v in pt[:2]] for pt in json.loads(points_json or "[]")]
    except (ValueError, TypeError):
        pts = []
    p1 = pts[0] if len(pts) > 0 else None
    p2 = pts[1] if len(pts) > 1 else None
    return (p1, p2, *_calib_labels(p1, p2))


def apply_calibration(selected_folders, p1, p2, real_dist_mm, manual_ppm):
    """Compute pixels_per_mm from the marked line (or use manual value) and save calibration.json.

    Points p1/p2 are in original-image pixels.
    """
    import math
    from core.paths import get_processed_folder
    from core.common import current_timestamp
    from pipeline.pixel_mass import save_calibration

    ppm = None
    if p1 and p2 and real_dist_mm and real_dist_mm > 0:
        ppm = math.dist(p1, p2) / real_dist_mm
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


def scan_legacy_collections_ui(dataset_root):
    """Scan *dataset_root* for legacy-format collections and populate the checkbox group."""
    if not dataset_root or not os.path.isdir(dataset_root):
        return (
            gr.update(choices=[], value=[], visible=False),
            gr.update(interactive=False),
            "No dataset folder selected. Pick a folder in the Setup tab first.\n",
        )

    results = Mothbot_LegacyConverter.scan_dataset_for_legacy(dataset_root)
    if not results:
        return (
            gr.update(choices=[], value=[], visible=False),
            gr.update(interactive=False),
            "✅ No legacy-format collections found — this dataset is already up to date.\n",
        )

    choices = []
    for info in results:
        label = (
            f"{info['rel_path']}  "
            f"({info['json_count']} JSON, {info['patch_count']} patches)"
        )
        choices.append((label, info["source_folder"]))

    return (
        gr.update(choices=choices, value=[c[1] for c in choices], visible=True),
        gr.update(interactive=True),
        f"Found {len(results)} legacy collection(s). Select which to convert and click Convert Selected.\n",
    )


def run_legacy_converter_ui(selected_folders, dataset_root, delete_originals):
    """Gradio generator that converts selected legacy collections."""
    SHOW_STOP = gr.update(visible=True, value="Stop Current Run", interactive=True)
    HIDE_STOP = gr.update(visible=False)

    if not selected_folders:
        yield "No collections selected.\n", HIDE_STOP
        return

    if not dataset_root or not os.path.isdir(dataset_root):
        yield "Dataset root not set — pick a folder in the Setup tab first.\n", HIDE_STOP
        return

    output_log = ""
    yield output_log, SHOW_STOP

    for folder in selected_folders:
        output_log += f"\n=== Converting: {folder} ===\n"
        yield output_log, SHOW_STOP
        try:
            for line in Mothbot_LegacyConverter.convert_collection(
                folder, dataset_root, delete_originals=bool(delete_originals)
            ):
                output_log += line
                yield output_log, SHOW_STOP
        except Exception as exc:
            output_log += f"❌ Exception: {exc}\n"
            yield output_log, SHOW_STOP

    output_log += "\n--- Conversion finished ---\n"
    yield output_log, HIDE_STOP


def run_pixel_mass_ui(selected_folders, pixels_per_mm, overwrite_nobg, overwrite_pixmass, model_name="birefnet-general", only_identified=False):
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
                model_name=model_name or "birefnet-general",
                only_identified=bool(only_identified),
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
            if _stop_requested():
                output_log += f"⛔ Pixel Mass stopped during {folder} — remaining collections skipped.\n"
                yield output_log, SHOW_STOP, COLLAPSE, NO_PREVIEW
                break
            output_log += f"✅ Pixel Mass completed for {folder}\n"
        except Exception as exc:
            _note_stage_error()
            output_log += f"\n❌ Exception: {exc}\n"
        yield output_log, SHOW_STOP, COLLAPSE, NO_PREVIEW

    output_log += "\n--- Pixel Mass finished ---"
    yield output_log, HIDE_STOP, EXPAND, NO_PREVIEW   # re-expand Step 1 when done


def get_index(selected_word):
    return TAXA_COLS.index(selected_word)


def run_detection_with_continue(selected_folders, yolo_model, imsz, overwrite_bot, delete_old_patches=False, external_keys=None):
    yolo_model = _resolve_model_path(yolo_model)
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
    from PIL import Image as PILImage

    # Slideshow state: cycle through patches from the last completed source image.
    slide_patches: list[str] = []
    slide_idx = 0

    def _open_slide(idx: int):
        try:
            return gr.update(value=PILImage.open(slide_patches[idx]), visible=True)
        except Exception:
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
                delete_old_model_patches=bool(delete_old_patches),
                dataset_root=dataset_root,
                tick_interval=0.3,
            ):
                if chunk is TICK:
                    # Advance slideshow while the next source image is being processed.
                    if slide_patches:
                        slide_idx = (slide_idx + 1) % len(slide_patches)
                        yield output_log, gr.update(interactive=False), SHOW_STOP, _open_slide(slide_idx)
                else:
                    output_log += chunk
                    # Collect patches emitted for the just-finished source image.
                    new_patches: list[str] = []
                    while True:
                        p = get_preview()
                        if p is None:
                            break
                        new_patches.append(p)
                    if new_patches:
                        slide_patches = new_patches
                        slide_idx = 0
                    yield output_log, gr.update(interactive=False), SHOW_STOP, (
                        _open_slide(slide_idx) if slide_patches else NO_IMG
                    )
            if _stop_requested():
                output_log += f"⛔ Detection stopped during {folder} — remaining collections skipped.\n"
                yield output_log, gr.update(interactive=False), SHOW_STOP, NO_IMG
                break
            output_log += f"✅ Detection completed for {folder}\n"
        except Exception as exc:
            had_error = True
            _note_stage_error()
            output_log += f"\n❌ Exception while processing {folder}: {exc}\n"
        yield output_log, gr.update(interactive=False), SHOW_STOP, NO_IMG

    output_log += "----------- Finished running Batch --------------"
    yield output_log, gr.update(interactive=(not had_error)), HIDE_STOP, NO_IMG


def _processed_dir_for(folder, dataset_root, is_external):
    """Where a collection's detection JSONs and patches live (without creating it)."""
    if is_external:
        return folder
    root = os.path.realpath(dataset_root or folder)
    rel = os.path.relpath(os.path.realpath(folder), root)
    return os.path.join(root, "_processed", "" if rel == "." else rel)


def load_blur_examples(selected_folders, threshold, max_json=300, max_unscored=200):
    """Sample patches with blurriness scores from the chosen collections for the ID-tab preview.

    Uses scores recorded by Detect/Cluster; for older datasets not yet scored,
    scores a limited sample on the fly (not written back — Cluster/ID do that).
    """
    import random
    from core.blur import blur_score as _blur_score, shape_needs_blur as _needs_blur

    examples = []
    unscored = []
    json_paths = []
    for entry in selected_folders or []:
        folder = entry["path"] if isinstance(entry, dict) else entry
        is_ext = entry.get("external", False) if isinstance(entry, dict) else False
        root = entry.get("dataset_root", folder) if isinstance(entry, dict) else folder
        base = _processed_dir_for(folder, root, is_ext)
        json_paths += glob.glob(os.path.join(base, "**", "*_botdetection.json"), recursive=True)
    random.Random(0).shuffle(json_paths)

    for json_path in json_paths[:max_json]:
        try:
            with open(json_path) as f:
                shapes = json.load(f).get("shapes", [])
        except Exception:
            continue
        for shape in shapes:
            patch = shape.get("patch_path")
            if not patch:
                continue
            patch_path = os.path.join(os.path.dirname(json_path), os.path.basename(patch))
            if not _needs_blur(shape):  # scored, and by the current method
                examples.append((float(shape["blur_score"]), patch_path))
            else:
                unscored.append(patch_path)

    for patch_path in unscored[:max_unscored]:
        image = cv2.imread(patch_path) if os.path.isfile(patch_path) else None
        if image is not None:
            examples.append((_blur_score(image), patch_path))

    examples.sort()
    image, caption = show_blur_example(examples, threshold)
    return examples, image, caption


def show_blur_example(examples, threshold):
    """Show the sampled patch whose blurriness is nearest the threshold."""
    if not examples:
        return None, "_No detection patches found in the chosen collections yet — run Detect first._"
    threshold = float(threshold)
    score, patch_path = min(examples, key=lambda e: abs(e[0] - threshold))
    if threshold >= 100:
        caption = (
            f"Threshold **100**: every patch is identified. "
            f"Showing a patch with blurriness **{score:.0f}**."
        )
    else:
        skipped = sum(1 for s, _ in examples if s > threshold)
        caption = (
            f"This example has blurriness **{score:.0f}**. At threshold **{threshold:.0f}**, "
            f"about **{100 * skipped / len(examples):.0f}%** of {len(examples):,} sampled patches "
            f"would be left unidentified."
        )
    return patch_path, caption


def run_ID(selected_folders, species_list, chosenrank, IDHum, IDBot, overwrite_bot, blur_threshold=100):
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
            "blur_threshold": blur_threshold,
        },
    )


def run_metadata(
    selected_folders,
    metadata_mode,
    metadata_csv,
    overwrite_existing,
    meta_manual_selected,
    mapping,
    external_keys,
    dataset_root,
    deployment_name,
    latitude,
    longitude,
    crew,
    project,
    site,
    habitat,
    device,
    firmware,
    utc,
    deployment_date,
    collect_date,
    attractor,
    attractor_location,
    height_above_ground,
    schedule,
    data_storage_location,
    notes,
):
    if metadata_mode == "Manual Entry":
        # Resolve the manual night-picker selection to actual folder paths.
        # Fall back to the global selected_folders if the picker is empty.
        if meta_manual_selected and mapping:
            target_folders, _ = confirm_selection(
                meta_manual_selected, mapping, external_keys or set(), dataset_root or ""
            )
        else:
            target_folders = selected_folders

        manual_meta = {
            "deployment_name": deployment_name or "",
            "latitude": latitude or "",
            "longitude": longitude or "",
            "crew": crew or "",
            "project": project or "",
            "site": site or "",
            "habitat": habitat or "",
            "device": device or "",
            "firmware": firmware or "",
            "UTC": utc or "",
            "deployment_date": deployment_date or "",
            "collect_date": collect_date or "",
            "attractor": attractor or "",
            "attractor_location": attractor_location or "",
            "height_above_ground": height_above_ground or "",
            "schedule": schedule or "",
            "data_storage_location": data_storage_location or "",
            "notes": notes or "",
        }
        yield from _run_batch_pipeline(
            selected_folders=target_folders,
            runner=Mothbot_InsertMetadata.run,
            start_message="---🔍 Running METADATA for {folder} ---\n",
            success_message="✅ Insert Metadata completed for {folder}\n",
            finish_message="------ Insert Metadata processing finished ------",
            kwargs_builder=lambda folder, dataset_root, _meta=manual_meta: {
                "input_path": folder,
                "manual_metadata": _meta,
                "dataset_root": dataset_root,
                "overwrite_existing": bool(overwrite_existing),
            },
        )
    else:
        yield from _run_batch_pipeline(
            selected_folders=selected_folders,
            runner=Mothbot_InsertMetadata.run,
            start_message="---🔍 Running METADATA for {folder} ---\n",
            success_message="✅ Insert Metadata completed for {folder}\n",
            finish_message="------ Insert Metadata processing finished ------",
            kwargs_builder=lambda folder, dataset_root: {
                "input_path": folder,
                "metadata_path": str(metadata_csv),
                "dataset_root": dataset_root,
                "overwrite_existing": bool(overwrite_existing),
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
            if _stop_requested():
                output_log += f"⛔ Cluster stopped during {folder} — remaining collections skipped.\n"
                yield output_log, gr.update(interactive=False), SHOW_STOP
                break
            output_log += f"✅ Cluster completed for {folder}\n"
        except Exception as exc:
            had_error = True
            _note_stage_error()
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
            if _stop_requested():
                output_log += f"⛔ Stopped during {folder} — remaining collections skipped.\n"
                yield output_log, SHOW_STOP
                break
            output_log += success_message.format(folder=folder)
        except Exception as exc:
            _note_stage_error()
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

DEFAULT_METADATA_CSV = ""


def _ensure_bundled_species_csv() -> str:
    """Return the path to the bundled worldwide species-list CSV, extracting it
    from the committed zip if the CSV hasn't been unpacked yet.

    The CSV is 269 MB (over GitHub's 100 MB limit) so only the zip is stored in
    git.  For a frozen app the zip is bundled inside sys._MEIPASS/specieslists/;
    the extracted CSV is written to ~/.mothbot/specieslists/ (writable on all
    platforms).  For source runs both the zip and the extracted CSV live in
    PROJECT_ROOT/specieslists/.
    """
    import zipfile

    zip_name = "Species_GBIF_Insecta_Worldwide_doi.org10.15468dl.hr7yfq.csv.zip"
    csv_name = zip_name[:-4]  # strip .zip

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        zip_path = Path(meipass) / "specieslists" / zip_name
        csv_dir  = Path.home() / ".mothbot" / "specieslists"
    else:
        zip_path = PROJECT_ROOT / "specieslists" / zip_name
        csv_dir  = PROJECT_ROOT / "specieslists"

    csv_path = csv_dir / csv_name

    if not zip_path.exists():
        return str(csv_path) if csv_path.exists() else ""

    # Re-extract if the unpacked CSV doesn't match the zip — e.g. it was
    # extracted before the bundled list was cleaned of non-insect species.
    if csv_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zipped_size = next(i.file_size for i in zf.infolist()
                                   if i.filename.endswith(".csv") and not i.filename.startswith("__"))
            if csv_path.stat().st_size == zipped_size:
                return str(csv_path)
            print("Bundled species list was updated — replacing the old extracted copy.")
        except Exception:
            return str(csv_path)

    try:
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting bundled worldwide species list — this happens once (~270 MB)...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only the CSV; skip macOS __MACOSX metadata entries
            for member in zf.namelist():
                if member.endswith(".csv") and not member.startswith("__"):
                    zf.extract(member, csv_dir)
                    extracted = csv_dir / member
                    if extracted != csv_path:
                        extracted.rename(csv_path)
                    break
        if csv_path.exists():
            print(f"Species list ready: {csv_path}")
            return str(csv_path)
    except Exception as exc:
        print(f"Warning: could not extract bundled species list: {exc}")
    return ""


DEFAULT_SPECIES_CSV = _ensure_bundled_species_csv()


def _discover_bundled_models():
    """Find MBD-*.pt files in trained_models/ for both packaged and source runs.

    Checks sys._MEIPASS first (PyInstaller bundle), then PROJECT_ROOT/trained_models.
    Returns a list of (label, path) tuples sorted newest-version-first, ready for
    a gr.Dropdown choices list.
    """
    search_dirs = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        search_dirs.append(Path(meipass) / "trained_models")
    search_dirs.append(PROJECT_ROOT / "trained_models")

    seen: set[str] = set()
    models = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pt in sorted(d.glob("MBD-*.pt")):
            m = re.match(r"MBD-(\d+)-(\d+)\.pt$", pt.name)
            if not m or pt.name in seen:
                continue
            seen.add(pt.name)
            major, minor = int(m.group(1)), int(m.group(2))
            models.append((major, minor, pt))

    models.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [
        (f"MBD-{major}-{minor} (bundled)", str(path.resolve()))
        for major, minor, path in models
    ]


BUNDLED_MODEL_CHOICES = _discover_bundled_models()
# Map label → path so we can resolve it even when Gradio sends the label string
# instead of the underlying value (a known gr.Dropdown allow_custom_value quirk).
BUNDLED_MODEL_LABEL_TO_PATH: dict[str, str] = {
    label: path for label, path in BUNDLED_MODEL_CHOICES
}
DEFAULT_YOLO_MODEL = BUNDLED_MODEL_CHOICES[0][1] if BUNDLED_MODEL_CHOICES else ""


def _resolve_model_path(value: str) -> str:
    """Return the real filesystem path for a model dropdown value.

    Gradio 5's gr.Dropdown with allow_custom_value=True can send the displayed
    label text instead of the underlying value when a bundled choice is selected.
    This maps the label back to the path; custom / already-absolute paths pass through.
    """
    return BUNDLED_MODEL_LABEL_TO_PATH.get(value, value)

demo = app()

if __name__ == "__main__":
    launch_kwargs = {"inbrowser": True}
    favicon = Path(__file__).with_name("favicon.png")
    if favicon.exists():
        launch_kwargs["favicon_path"] = str(favicon)
    ensure_single_instance(url="http://127.0.0.1:7860")
    start_tray(url="http://127.0.0.1:7860")
    demo.launch(**launch_kwargs)