"""
Thread-safe single-slot channel for streaming live preview images
from the detection pipeline to the Gradio UI.

detect.py calls emit_preview() with a patch path; the Gradio runner
calls get_preview() at each yield boundary to update the image component.
"""

import queue

_q: queue.Queue = queue.Queue(maxsize=1)


def emit_preview(path: str) -> None:
    """Push a preview image path. Replaces any unseen previous value."""
    try:
        _q.put_nowait(path)
    except queue.Full:
        try:
            _q.get_nowait()
        except queue.Empty:
            pass
        _q.put_nowait(path)


def get_preview() -> str | None:
    """Return the latest preview path, or None if nothing is waiting."""
    try:
        return _q.get_nowait()
    except queue.Empty:
        return None


def clear_preview() -> None:
    """Drain any pending previews (call before starting a new detection run)."""
    while True:
        try:
            _q.get_nowait()
        except queue.Empty:
            break
