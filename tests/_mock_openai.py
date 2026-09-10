"""Minimal OpenAI-compatible /v1/responses server for offline
verification of the book's adapter code."""
import json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def build_response(text, model, in_tok=42, out_tok=None):
    out_tok = out_tok if out_tok is not None else len(text.split())
    return {
        "id": "resp_mock_1", "object": "response",
        "created_at": time.time(), "model": model,
        "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto",
        "tools": [],
        "output": [{
            "id": "msg_1", "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text,
                         "annotations": []}],
        }],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                  "total_tokens": in_tok + out_tok,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append(body)
        model = body.get("model", "mock-model")
        text = "billing"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            resp = build_response(text, model)
            def send(ev, data):
                self.wfile.write(f"event: {ev}\n".encode())
                self.wfile.write(
                    f"data: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            send("response.created", {
                "type": "response.created", "sequence_number": 0,
                "response": build_response("", model)})
            for i, piece in enumerate(["bil", "ling"], start=1):
                send("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "sequence_number": i, "item_id": "msg_1",
                    "output_index": 0, "content_index": 0,
                    "logprobs": [], "delta": piece})
            send("response.completed", {
                "type": "response.completed",
                "sequence_number": 9, "response": resp})
            return
        payload = json.dumps(build_response(text, model)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

SEEN = []
def serve(port=8099):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
