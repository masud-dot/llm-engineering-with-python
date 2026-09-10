"""A /v1/responses endpoint that asks for a tool call."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = []
STATE = {"mode": "call"}

def _envelope(output):
    return json.dumps({
        "id": "r", "object": "response", "created_at": 0,
        "model": "mock-model", "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto",
        "tools": [], "output": output,
        "usage": {"input_tokens": 60, "output_tokens": 25,
                  "total_tokens": 85,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {
                      "reasoning_tokens": 0}},
    }).encode()

CALL = [{
    "type": "function_call", "id": "fc_1", "call_id": "call_abc",
    "name": "search_docs", "status": "completed",
    "arguments": '{"query": "refund window", "limit": 3}',
}]
TEXT = [{
    "id": "m", "type": "message", "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text",
                 "text": "Refunds take fourteen days.",
                 "annotations": []}],
}]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        SEEN.append(json.loads(self.rfile.read(n) or b"{}"))
        output = CALL if STATE["mode"] == "call" else TEXT
        payload = _envelope(output)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def serve(port=8092):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
