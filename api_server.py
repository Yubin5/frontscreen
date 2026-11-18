"""Screen flattening API: send a photo, get the display screen back as a front-view JPEG (and its text).

  POST /flatten   body = photo bytes (JPEG/PNG/WebP)  ->  200 image/jpeg
  POST /read      body = photo bytes  ->  200 JSON: text read from the flattened screen (OCR)
  GET  /health    {"ok": true|false}  (is the SAM/detector server up?)

Spec: docs/api.md. Needs the SAM/detector server (sam/sam_server.py, :8791).

Env:
  HOSTS         comma-separated bind addresses (default 127.0.0.1)
  PORT          default 8793
  API_KEY_FILE  default secrets/api_key; when the file exists, callers not on loopback must send
                "Authorization: Bearer <key>" (or "X-API-Key: <key>")
"""
import base64, hmac, ipaddress, json, os, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import rectify, screen_ocr  # noqa: E402

MAX_BYTES = 20 * 1024 * 1024
API_KEY_FILE = os.environ.get("API_KEY_FILE", os.path.join(LAB, "secrets", "api_key"))
SAM_HEALTH = os.environ.get("SAM_HEALTH", "http://127.0.0.1:8791/health")
# rectify keeps the last SAM response in a module attribute (crop.sam.last), so detections run one at a time;
# the SAM server is single-threaded anyway, so this costs no throughput
_detect_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def _api_key():
    try:
        return open(API_KEY_FILE, encoding="utf-8").read().strip()
    except FileNotFoundError:
        return ""


class Handler(BaseHTTPRequestHandler):
    server_version = "flatten-api/2"

    def _send(self, status, body, ctype, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _auth(self):
        key = _api_key()
        if not key:
            return
        try:
            if ipaddress.ip_address(self.client_address[0]).is_loopback:
                return
        except ValueError:
            pass
        given = self.headers.get("X-API-Key", "")
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        if not given or not hmac.compare_digest(given.encode(), key.encode()):
            raise ApiError(401, "missing or wrong API key (Authorization: Bearer <key>)")

    def do_GET(self):
        try:
            self._auth()
            if urlparse(self.path).path != "/health":
                raise ApiError(404, "unknown endpoint (POST /flatten, POST /read, GET /health)")
            try:
                with urllib.request.urlopen(SAM_HEALTH, timeout=2.0) as r:
                    ok = bool(json.load(r).get("ok"))
            except Exception:  # noqa: BLE001 - any failure means "not reachable"
                ok = False
            self._json(200, {"ok": ok})
        except ApiError as e:
            self._json(e.status, {"error": e.message})

    def do_POST(self):
        t0 = time.time()
        try:
            self._auth()
            u = urlparse(self.path)
            if u.path not in ("/flatten", "/read"):
                raise ApiError(404, "unknown endpoint (POST /flatten, POST /read, GET /health)")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                raise ApiError(400, "empty body: send the photo bytes")
            if n > MAX_BYTES:
                raise ApiError(413, "photo over 20 MB")
            data = self.rfile.read(n)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ApiError(400, "could not decode the image (JPEG/PNG/WebP expected)")
            q = parse_qs(u.query)
            try:
                max_side = int(q.get("max_side", ["1600"])[0])
            except ValueError:
                max_side = 0
            if not 256 <= max_side <= 4096:
                raise ApiError(400, "max_side must be an integer between 256 and 4096")
            if u.path == "/read":
                max_side = 1600  # OCR is tuned for this size; the boxes refer to this image
            jpeg = data if data[:2] == b"\xff\xd8" else cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            try:
                with _detect_lock:
                    d = rectify.detect(img, jpeg)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                raise ApiError(503, f"SAM/detector server not reachable: {e}")
            if not d.get("quad"):
                raise ApiError(422, "no display screen found in the photo")
            flat, _ = rectify.warp(img, d["quad"], aspect=True, long_side=max_side)
            ok = (d.get("validation") or {}).get("ok", True)
            if u.path == "/read":
                o = screen_ocr.read(flat)
                res = {"text": o["text"], "lines": o["lines"], "check": "ok" if ok else "doubtful",
                       "corners": [[round(x, 4), round(y, 4)] for x, y in d["quad"]],
                       "image_size": [flat.shape[1], flat.shape[0]],
                       "ms": {"ocr": o["ms"], "total": round((time.time() - t0) * 1000)}}
                if q.get("image", ["0"])[0] == "1":
                    res["image"] = base64.b64encode(cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()
                return self._json(200, res)
            out = cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            self._send(200, out, "image/jpeg", {
                "X-Screen-Check": "ok" if ok else "doubtful",  # doubtful: worth letting the user confirm
                "X-Screen-Corners": ";".join(f"{x:.4f},{y:.4f}" for x, y in d["quad"]),
                "X-Process-Ms": str(round((time.time() - t0) * 1000)),
            })
        except ApiError as e:
            self._json(e.status, {"error": e.message})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s %s\n" % (self.address_string(), fmt % args))


def main():
    port = int(os.environ.get("PORT", 8793))
    servers = []
    for host in os.environ.get("HOSTS", "127.0.0.1").split(","):
        try:
            servers.append(ThreadingHTTPServer((host.strip(), port), Handler))
        except OSError as e:
            print(f"[api] cannot bind {host}:{port}: {e}", flush=True)
    if not servers:
        sys.exit(1)
    if os.environ.get("OCR_PRELOAD", "1") == "1":
        t = time.time(); screen_ocr.warm(); print(f"[api] OCR models loaded in {time.time() - t:.1f}s", flush=True)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("[api] flatten API on " + ", ".join(f"{s.server_address[0]}:{port}" for s in servers)
          + (" (API key required off-loopback)" if _api_key() else " (no API key file: open)"), flush=True)
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
