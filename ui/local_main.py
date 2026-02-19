#!/usr/bin/env python3

import os
from pathlib import Path

from ui.app import get_demo


def main():
    server_port = os.getenv("GRADIO_SERVER_PORT")
    launch_kwargs = {
        "inbrowser": os.getenv("MOTHBOT_INBROWSER", "1") == "1",
        "share": False,
    }
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    favicon = Path(__file__).resolve().parent.parent / "assets" / "favicon.png"
    if favicon.exists():
        launch_kwargs["favicon_path"] = str(favicon)
    get_demo().launch(**launch_kwargs)


if __name__ == "__main__":
    main()

