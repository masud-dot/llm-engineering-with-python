"""An endpoint that fails on demand, for reliability tests."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"fail_next": 0, "status": 503, "retry_after": None,
         "calls": 0}

def _body(text="ok"):
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
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {
                      "reasoning_tokens": 0}},
    }).encode()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        STATE["calls"] += 1
        if STATE["fail_next"] > 0:
            STATE["fail_next"] -= 1
            payload = json.dumps({"error": {
                "message": "injected failure",
                "type": "server_error"}}).encode()
            self.send_response(STATE["status"])
            self.send_header("Content-Type", "application/json")
            if STATE["retry_after"] is not None:
                self.send_header(
                    "retry-after", str(STATE["retry_after"]))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = _body()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def serve(port=8096):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
