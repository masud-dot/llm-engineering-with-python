"""Record real API traffic once, replay it forever.

Recording talks to a real endpoint through base_url and
writes each response to disk. Replaying serves those files
with no network at all, so the suite is hermetic.
"""
import hashlib
import json
import pathlib
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CASSETTES = pathlib.Path("tests/cassettes")


def key_for(path: str, body: dict) -> str:
    """Identify a request by route and its meaningful fields."""
    parts = {
        "path": path,
        "model": body.get("model"),
        "input": body.get("input"),
        "instructions": body.get("instructions"),
        "messages": body.get("messages"),
    }
    blob = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class MissingCassette(RuntimeError):
    """Replay was asked for a request never recorded."""


def _handler(mode: str, upstream: str | None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            body = json.loads(raw or b"{}")
            name = key_for(self.path, body)
            path = CASSETTES / f"{name}.json"
            if mode == "replay":
                if not path.exists():
                    self._fail(
                        404,
                        f"no cassette {name} for {self.path}",
                    )
                    return
                payload = path.read_bytes()
            else:
                assert upstream is not None
                request = urllib.request.Request(
                    upstream + self.path,
                    data=raw,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request) as reply:
                    payload = reply.read()
                CASSETTES.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _fail(self, code: int, message: str):
            payload = json.dumps(
                {"error": {"message": message}}
            ).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return H


def serve(port: int, mode: str = "replay", upstream=None):
    """Start a cassette server. mode is 'record' or 'replay'."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), _handler(mode, upstream)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
