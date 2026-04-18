"""
tray.py – System tray icon for Mothbot.

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


def _find_favicon() -> Path | None:
    """
    Locate favicon.png whether running from source or a PyInstaller bundle.
    PyInstaller extracts bundled files to sys._MEIPASS at runtime.
    """
    candidates = [
        # PyInstaller bundle: assets/ is extracted next to the exe
        Path(getattr(sys, "_MEIPASS", "")) / "assets" / "favicon.png",
        # Running from source: apps/assets/favicon.png (two levels up from ui/)
        Path(__file__).resolve().parent.parent / "assets" / "favicon.png",
        # Fallback: favicon.png sitting right next to tray.py
        Path(__file__).with_name("favicon.png"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_icon_image(icon_path: Path | None = None, size: int = 64) -> "Image.Image":
    """Return a PIL Image for the tray icon."""
    path = icon_path or _find_favicon()
    if path and path.exists():
        return Image.open(path).convert("RGBA").resize((size, size))

    # Fallback: draw a simple green moth shape
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    draw.ellipse([cx - 6, cy - 14, cx + 6, cy + 14], fill=(50, 180, 80, 255))
    draw.ellipse([4, cy - 12, cx - 2, cy + 10], fill=(80, 200, 100, 220))
    draw.ellipse([cx + 2, cy - 12, size - 4, cy + 10], fill=(80, 200, 100, 220))
    return img


def start_tray(url: str = "http://127.0.0.1:7861", icon_path: Path | None = None) -> None:
    """
    Spawn the system-tray icon in a daemon thread.

    Parameters
    ----------
    url:
        The local URL Gradio is listening on.
    icon_path:
        Optional explicit path to a PNG icon. If None, auto-detected.
    """
    if not _PYSTRAY_AVAILABLE:
        print(
            "[tray] pystray or Pillow not installed – tray icon disabled.\n"
            "       Run:  pip install pystray Pillow"
        )
        return

    thread = threading.Thread(target=_run_tray, args=(url, icon_path), daemon=True)
    thread.start()


def _run_tray(url: str, icon_path: Path | None) -> None:
    icon_image = _load_icon_image(icon_path)

    def on_open(icon, item):
        webbrowser.open(url)

    def on_quit(icon, item):
        icon.stop()
        threading.Timer(0.5, lambda: os._exit(0)).start()

    menu = pystray.Menu(
        pystray.MenuItem("Open Mothbot", on_open, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon(
        name="Mothbot",
        icon=icon_image,
        title="Mothbot (running)",
        menu=menu,
    )
    icon.run()
