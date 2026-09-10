"""A /v1/responses endpoint that replies from a script."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = []
STATE = {"replies": []}

def _body(text):
    return json.dumps({
        "id": "r", "object": "response", "created_at": 0,
        "model": "mock-model", "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto",
        "tools": [],
        "output": [{"id": "m", "type": "message",
                    "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text",
                                 "text": text,
                                 "annotations": []}]}],
        "usage": {"input_tokens": 50, "output_tokens": 30,
                  "total_tokens": 80,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {
                      "reasoning_tokens": 0}},
    }).encode()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        SEEN.append(json.loads(self.rfile.read(n) or b"{}"))
        replies = STATE["replies"]
        text = replies.pop(0) if replies else "{}"
        payload = _body(text)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def serve(port=8094):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
