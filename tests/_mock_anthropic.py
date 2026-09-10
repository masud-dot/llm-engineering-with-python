"""Minimal Anthropic-compatible /v1/messages server."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        SEEN.append(json.loads(self.rfile.read(n) or b"{}"))
        payload = json.dumps({
            "id": "msg_mock", "type": "message",
            "role": "assistant", "model": "mock-model",
            "content": [{"type": "text", "text": "billing"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 37, "output_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def serve(port=8098):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
