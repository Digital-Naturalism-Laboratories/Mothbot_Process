"""Local page for rating patch blurriness by hand (ground truth for checking core/blur.py).

    python tools/blur_calibration/rate.py [data_dir] [port]

Shows the patches listed in <data_dir>/sample.json (from make_sample.py) one at a
time and saves each rating to <data_dir>/labels.json as you go, so you can stop
and resume. Only the listed patches are served, and the page never sees their
paths, datasets or current scores. data_dir defaults to ~/.mothbot/blur_calibration.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.mothbot/blur_calibration")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
SAMPLE = json.load(open(os.path.join(DATA_DIR, "sample.json")))
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")


def load_labels():
    try:
        with open(LABELS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        elif self.path == "/sample":
            self._send(json.dumps([{"i": r["i"], "w": r["w"], "h": r["h"]} for r in SAMPLE]).encode(), "application/json")
        elif self.path == "/labels":
            self._send(json.dumps(load_labels()).encode(), "application/json")
        elif self.path.startswith("/img/"):
            try:
                with open(SAMPLE[int(self.path[5:])]["path"], "rb") as f:
                    self._send(f.read(), "image/jpeg")
            except Exception:
                self._send(b"not found", "text/plain", 404)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/label":
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        labels = load_labels()
        labels[str(int(body["i"]))] = {"rating": body["rating"], "cause": body.get("cause")}
        tmp = LABELS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(labels, f, indent=1)
        os.replace(tmp, LABELS_PATH)  # never leave a half-written labels file
        self._send(json.dumps({"saved": len(labels)}).encode(), "application/json")


if __name__ == "__main__":
    print(f"Blur rating page: http://localhost:{PORT}  ({len(SAMPLE)} patches; ratings -> {LABELS_PATH})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
