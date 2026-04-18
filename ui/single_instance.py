"""
single_instance.py – Prevent multiple Mothbot processes running at once.

Stores both PID and the actual server URL in the lock file so the second
launch can open the correct browser tab regardless of which port Gradio
ended up on.

No extra dependencies – stdlib only.
"""

import os
import sys
import tempfile
import webbrowser
import atexit
from pathlib import Path

_LOCK_FILE = Path(tempfile.gettempdir()) / "mothbot.lock"


def _pid_is_running(pid: int) -> bool:
    """Return True if a process with *pid* exists on this machine."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle == 0:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _remove_lock() -> None:
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def ensure_single_instance(url: str = "http://127.0.0.1:7861") -> None:
    """
    Call this once at startup, BEFORE ``demo.launch()``.

    * If no other instance is running: writes a lock file with the current
      PID + URL, and registers a cleanup hook to remove it on exit.
    * If another instance is already running: opens the URL stored in the
      lock file (i.e. the port the first instance actually bound to),
      then exits immediately.
    """
    if _LOCK_FILE.exists():
        try:
            pid_str, stored_url = _LOCK_FILE.read_text().strip().splitlines()
            stored_pid = int(pid_str)
        except (ValueError, OSError):
            stored_pid = 0
            stored_url = url

        if _pid_is_running(stored_pid):
            print(
                f"[single_instance] Mothbot already running (PID {stored_pid}).\n"
                f"                  Opening existing instance: {stored_url}"
            )
            webbrowser.open(stored_url)
            sys.exit(0)
        else:
            # Stale lock from a previous crash – clean up and continue.
            _remove_lock()

    # First instance: claim the lock with PID on line 1, URL on line 2.
    _LOCK_FILE.write_text(f"{os.getpid()}\n{url}")
    atexit.register(_remove_lock)
