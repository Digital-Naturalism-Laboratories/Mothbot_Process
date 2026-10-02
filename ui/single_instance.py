"""
single_instance.py – Prevent multiple Mothbot processes running at once.

Stores both PID and the actual server URL in the lock file so the second
launch can open the correct browser tab regardless of which port Gradio
ended up on.
"""

import os
import socket
import sys
import tempfile
import time
import webbrowser
import atexit
from pathlib import Path
from urllib.parse import urlparse

_LOCK_FILE = Path(tempfile.gettempdir()) / "mothbot.lock"

# How long a second launch waits for a first copy that is still starting up
# (the CUDA build can take a couple of minutes on a cold start).
_WAIT_FOR_STARTUP_SECONDS = 300


def _pid_is_running(pid: int) -> bool:
    """Return True if a process with *pid* exists on this machine."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle == 0:
            return False
        try:
            # A process that has exited can still be opened while anything holds a
            # handle to it, so check that it is actually still running.
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _is_mothbot_process(pid: int) -> bool:
    """True if *pid* is alive and is this same program (not a stale lock whose
    process ID the OS has since given to something else)."""
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        import psutil
    except ImportError:
        return _pid_is_running(pid)
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        try:
            same = lambda path: os.path.normcase(os.path.realpath(path))  # noqa: E731
            return same(proc.exe()) == same(sys.executable)
        except (psutil.AccessDenied, OSError):
            return True  # alive, but we may not inspect it — assume it's ours
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _server_answers(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


def release_lock() -> None:
    """Remove the lock file. Call before any hard exit (os._exit / a kill),
    which skips atexit — a left-over lock made the next launch think Mothbot
    was still running."""
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass



def ensure_single_instance(url: str = "http://127.0.0.1:7861") -> None:
    """
    Call this once at startup, BEFORE ``demo.launch()``.

    * If no other instance is running: writes a lock file with the current
      PID + URL, and registers a cleanup hook to remove it on exit.
    * If another instance is running: waits until its server answers (it may
      still be starting up), opens its URL, then exits. If that instance turns
      out to be gone, this launch carries on as the first instance.
    """
    if _LOCK_FILE.exists():
        try:
            pid_str, stored_url = _LOCK_FILE.read_text().strip().splitlines()
            stored_pid = int(pid_str)
        except (ValueError, OSError):
            stored_pid = 0
            stored_url = url

        deadline = time.monotonic() + _WAIT_FOR_STARTUP_SECONDS
        while _is_mothbot_process(stored_pid):
            if _server_answers(stored_url) or time.monotonic() > deadline:
                print(
                    f"[single_instance] Mothbot already running (PID {stored_pid}).\n"
                    f"                  Opening existing instance: {stored_url}"
                )
                webbrowser.open(stored_url)
                sys.exit(0)
            time.sleep(1)  # still starting up: open the browser once it's ready

        # Stale lock from a previous run (crash, hard quit) – clean up and continue.
        release_lock()

    # First instance: claim the lock with PID on line 1, URL on line 2.
    _LOCK_FILE.write_text(f"{os.getpid()}\n{url}")
    atexit.register(release_lock)
