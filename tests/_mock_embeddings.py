"""A deterministic /v1/embeddings endpoint."""
import hashlib, json, math, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = []
DIMS = 8

def _vector(text, dims):
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] / 255.0 - 0.5
           for i in range(dims)]
    length = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / length for v in raw]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append(body)
        texts = body["input"]
        if isinstance(texts, str):
            texts = [texts]
        dims = body.get("dimensions", DIMS)
        payload = json.dumps({
            "object": "list", "model": body.get("model", "mock"),
            "data": [{"object": "embedding", "index": i,
                      "embedding": _vector(t, dims)}
                     for i, t in enumerate(texts)],
            "usage": {"prompt_tokens": sum(len(t.split())
                                           for t in texts),
                      "total_tokens": 0},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def serve(port=8093):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
