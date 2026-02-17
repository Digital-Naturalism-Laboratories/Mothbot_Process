#!/usr/bin/env python3

from pathlib import Path

from ui.app import get_demo


def main():
    launch_kwargs = {"inbrowser": True}
    favicon = Path(__file__).resolve().parent.parent / "assets" / "favicon.png"
    if favicon.exists():
        launch_kwargs["favicon_path"] = str(favicon)
    get_demo().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
