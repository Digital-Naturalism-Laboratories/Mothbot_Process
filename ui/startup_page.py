"""
startup_page.py – A "Mothbot is starting…" page shown while the app loads.

Loading the AI libraries can take minutes (the Windows CUDA build took ~5 min
after a restart), and until Gradio is listening the browser shows a connection
error — or nothing happens at all, so people double-click again. A tiny stdlib
web server answers on Mothbot's own address with a waiting page; the page polls
and turns into the app by itself once Gradio has taken the address over.
"""

import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STARTING_HEADER = "X-Mothbot-Starting"

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mothbot is starting…</title>
<style>
  :root { color-scheme: light dark; --bg: #f6f5f2; --card: #ffffff; --ink: #1f1f1f; --muted: #6b6b6b; --accent: #e8590c; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161616; --card: #222222; --ink: #ececec; --muted: #a0a0a0; } }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg);
         color: var(--ink); font: 16px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 440px; margin: 16px; padding: 32px; background: var(--card); border-radius: 14px;
         box-shadow: 0 6px 30px rgba(0,0,0,.12); text-align: center; }
  h1 { font-size: 22px; margin: 12px 0 8px; }
  p { color: var(--muted); margin: 8px 0; }
  .spinner { width: 40px; height: 40px; margin: 0 auto; border-radius: 50%;
             border: 4px solid color-mix(in srgb, var(--accent) 25%, transparent);
             border-top-color: var(--accent); animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #elapsed { font-variant-numeric: tabular-nums; color: var(--ink); }
  #slow { display: none; font-size: 14px; }
  code { font-size: 13px; word-break: break-all; }
</style>
</head>
<body>
<main>
  <div class="spinner" aria-hidden="true"></div>
  <h1>🦋 Mothbot is starting…</h1>
  <p>Loading the AI models. This can take a few minutes, especially the first time after restarting the computer.</p>
  <p>This page opens Mothbot by itself when it's ready — no need to click the Mothbot icon again.</p>
  <p>Waiting <span id="elapsed">0:00</span></p>
  <p id="slow">Still loading after 10 minutes? Something may have gone wrong — the log is at<br><code>__LOG_PATH__</code></p>
</main>
<script>
  const started = Date.now();
  setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    document.getElementById('elapsed').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
    if (s >= 600) document.getElementById('slow').style.display = 'block';
  }, 1000);
  async function check() {
    try {
      const response = await fetch(location.href, { cache: 'no-store' });
      if (response.ok && !response.headers.get('__HEADER__')) { location.reload(); return; }
    } catch (e) { /* the waiting page has stopped and the app isn't listening yet */ }
    setTimeout(check, 1500);
  }
  setTimeout(check, 1500);
</script>
</body>
</html>
"""


class StartupPage:
    """Serves the waiting page on 127.0.0.1:*port* until stop() is called."""

    def __init__(self, port: int, log_path: str = ""):
        body = _PAGE.replace("__HEADER__", STARTING_HEADER).replace("__LOG_PATH__", log_path or "the Mothbot logs folder")
        page = body.encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header(STARTING_HEADER, "1")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *args):  # stdio may be absent in the windowed app
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Free the port so Gradio can take it; the open page keeps polling until the app answers."""
        self._server.shutdown()
        self._server.server_close()


def show_startup_page(port: int, log_path: str = "") -> "StartupPage | None":
    """Start the waiting page and open it in the browser. Returns None (and opens
    nothing) if the page can't be served, so the app falls back to opening the
    browser when Gradio is ready."""
    try:
        page = StartupPage(port, log_path)
    except OSError:
        return None
    webbrowser.open(f"http://127.0.0.1:{port}")
    return page
