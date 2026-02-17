"""Primary UI module accessors."""

from ui import gradio_app


def get_demo():
    """Return the Gradio demo instance."""
    return gradio_app.demo

