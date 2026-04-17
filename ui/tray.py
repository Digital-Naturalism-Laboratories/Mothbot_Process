"""
tray.py – System tray icon for Mothbot.

Adds a tray icon (Windows / macOS / Linux) so users can see the app is
running and quit it cleanly instead of leaving orphaned processes.

Dependencies (add to pyproject.toml / requirements.txt):
    pystray>=0.19
    Pillow>=10.0
"""

import os
import sys
import threading
import webbrowser
from pathlib import Path

try:
    import pystray
    from PIL import Image, ImageDraw
    _PYSTRAY_AVAILABLE = True
except ImportError:
    _PYSTRAY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
#  Icon image helpers
# ──────────────────────────────────────────────────────────────

def _load_icon_image(size: int = 64) -> "Image.Image":
    """
    Return a PIL Image for the tray icon.
    Prefers a favicon.png sitting next to this file; falls back to a
    simple green moth-ish shape drawn with Pillow.
    """
    favicon = Path(__file__).with_name("favicon.png")
    if favicon.exists():
        img = Image.open(favicon).convert("RGBA").resize((size, size))
        return img

    # Fallback: draw a minimal icon
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Body
    cx, cy = size // 2, size // 2
    draw.ellipse([cx - 6, cy - 14, cx + 6, cy + 14], fill=(50, 180, 80, 255))
    # Wings
    draw.ellipse([4, cy - 12, cx - 2, cy + 10], fill=(80, 200, 100, 220))
    draw.ellipse([cx + 2, cy - 12, size - 4, cy + 10], fill=(80, 200, 100, 220))
    return img


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def start_tray(url: str = "http://127.0.0.1:7860") -> None:
    """
    Spawn the system-tray icon in a daemon thread.

    Call this *before* ``demo.launch()`` so the icon appears as soon as
    the app starts.  It is safe to call even if pystray is not installed –
    it will simply log a warning and return.

    Parameters
    ----------
    url:
        The local URL Gradio is listening on.  Passed to the
        "Open Mothbot" menu item so clicking it re-opens the browser tab.
    """
    if not _PYSTRAY_AVAILABLE:
        print(
            "[tray] pystray or Pillow not installed – tray icon disabled.\n"
            "       Run:  pip install pystray Pillow"
        )
        return

    thread = threading.Thread(target=_run_tray, args=(url,), daemon=True)
    thread.start()


# ──────────────────────────────────────────────────────────────
#  Internal
# ──────────────────────────────────────────────────────────────

def _run_tray(url: str) -> None:
    """Build and run the tray icon (blocking – must be in its own thread)."""

    icon_image = _load_icon_image()

    def on_open(icon, item):          # noqa: ARG001
        webbrowser.open(url)

    def on_quit(icon, item):          # noqa: ARG001
        icon.stop()
        # Give Gradio a moment to finish any in-flight requests, then exit.
        threading.Timer(0.5, lambda: os._exit(0)).start()

    menu = pystray.Menu(
        pystray.MenuItem("Open Mothbot", on_open, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon(
        name="Mothbot",
        icon=icon_image,
        title="Mothbot (running)",   # tooltip on hover
        menu=menu,
    )
    icon.run()
