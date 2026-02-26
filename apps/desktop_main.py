#!/usr/bin/env python3

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import sys
import traceback


_DEVNULL_STREAMS = []


def _ensure_stdio_streams() -> None:
    """
    PyInstaller windowed mode can leave stdio streams as None on Windows.
    Uvicorn's default logging formatter expects streams with isatty().
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            fallback = open(os.devnull, "w", encoding="utf-8")
            _DEVNULL_STREAMS.append(fallback)
            setattr(sys, stream_name, fallback)


def _log_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / "Mothbot" / "desktop.log"
    if sys.platform.startswith("win"):
        appdata = Path.home() / "AppData" / "Local"
        return appdata / "Mothbot" / "logs" / "desktop.log"
    return home / ".local" / "state" / "mothbot" / "desktop.log"


def _configure_logging() -> Path:
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root.addHandler(handler)
    return log_path


def _install_exception_hooks() -> None:
    def _handle_exception(exc_type, exc_value, exc_tb):
        logging.getLogger("mothbot.desktop").critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _handle_exception


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_server_port(preferred_port: int = 7861) -> int:
    override = os.getenv("GRADIO_SERVER_PORT")
    if override:
        try:
            return int(override)
        except ValueError:
            logging.getLogger("mothbot.desktop").warning(
                "Ignoring invalid GRADIO_SERVER_PORT value: %r", override
            )

    if _is_port_available(preferred_port):
        return preferred_port

    # Fallback to a random ephemeral port when 7861 is occupied.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main():
    _ensure_stdio_streams()
    log_path = _configure_logging()
    _install_exception_hooks()
    logger = logging.getLogger("mothbot.desktop")
    logger.info("Mothbot desktop startup")
    logger.info("Python executable: %s", sys.executable)
    logger.info("Log file: %s", log_path)

    server_port = _pick_server_port(preferred_port=7861)
    launch_kwargs = {
        "inbrowser": True,
        "server_name": "127.0.0.1",
        "server_port": server_port,
    }
    logger.info("Using server port: %s", server_port)
    favicon = Path(__file__).resolve().parent.parent / "assets" / "favicon.png"
    if favicon.exists():
        launch_kwargs["favicon_path"] = str(favicon)

    try:
        from ui.app import get_demo

        logger.info("Launching Gradio app")
        get_demo().launch(**launch_kwargs)
    except Exception:
        logger.error("Desktop startup failed", exc_info=True)
        # Keep this explicit trace write in case logging handler fails unexpectedly.
        try:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write("\n--- UNHANDLED STARTUP EXCEPTION ---\n")
                traceback.print_exc(file=fp)
                fp.write("--- END STARTUP EXCEPTION ---\n")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
