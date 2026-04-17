"""
single_instance.py – Prevent multiple Mothbot processes running at once.

Uses a PID lock file in the system temp directory.  If a lock file exists
and the recorded PID is still alive, the second launch opens the browser
tab of the already-running instance and exits immediately.

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
        # os.kill(pid, 0) is unreliable on Windows – use the Win32 API instead.
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle == 0:
            return False  # process doesn't exist
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    else:
        try:
            # os.kill(pid, 0) doesn't kill anything – it just checks existence.
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False      # process is gone
        except PermissionError:
            return True       # exists but owned by another user


def _remove_lock() -> None:
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def ensure_single_instance(url: str = "http://127.0.0.1:7860") -> None:
    """
    Call this once at startup, before ``demo.launch()``.

    * If no other instance is running: writes a lock file with the current
      PID and registers a cleanup hook so the file is removed on exit.
    * If another instance is already running: opens *url* in the default
      browser so the user lands on the existing UI, then exits immediately.
    """
    if _LOCK_FILE.exists():
        try:
            stored_pid = int(_LOCK_FILE.read_text().strip())
        except (ValueError, OSError):
            stored_pid = 0

        if _pid_is_running(stored_pid):
            print(
                f"[single_instance] Mothbot is already running (PID {stored_pid}).\n"
                f"                  Opening existing instance in browser: {url}"
            )
            webbrowser.open(url)
            sys.exit(0)
        else:
            # Stale lock left over from a previous crash – clean it up.
            _remove_lock()

    # We are the first instance – claim the lock.
    _LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(_remove_lock)
