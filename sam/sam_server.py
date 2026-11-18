"""SAM screen refiner, GPU (Arc) with VRAM safety.

POST /screen?box=x1,y1,x2,y2&pos=x,y;x,y&neg=x,y;...   (fractions 0-1)   body = image
  box = rough screen box (e.g. from the VLM); pos/neg = points that must / must not be on the screen
  -> {quad: [[x,y]*4] fractions tl,tr,br,bl | null, score, area, ms, device, embed_cached, note}
POST /detect   body = image
  open-vocabulary detector (Grounding DINO tiny) for the screen's rough box, used instead of the VLM guess
  -> {box: [x1,y1,x2,y2] fractions | null, score, boxes: [{box, score}] (top 3, best first), ms, device, note}
GET  /health -> device state, VRAM headroom, last fallback

VRAM safety (the GPU may be shared with other apps):
  * the fp32 master copy always lives on the CPU; a bf16 copy is put on the GPU only when needed
  * before using the GPU, ask vram-guard (127.0.0.1:8792) for host-wide free VRAM; below
    MIN_FREE_GIB -> run this request on the CPU instead
  * out-of-memory on the GPU -> free it, rerun on the CPU, stay on the CPU for OOM_COOLDOWN_S
  * no request for IDLE_OFFLOAD_S -> drop the GPU copy so the VRAM is free for the other apps
"""
import copy, hashlib, json, os, time, threading, urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

MODEL = os.environ.get("SAM_MODEL", "facebook/sam-vit-base")
GPU = os.environ.get("DEVICE", "cpu")                      # "xpu" = Intel Arc; "cpu" disables the GPU path
# fp16 overflows to NaN in SAM on the Arc; bf16 keeps fp32's range at half the memory
GPU_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("DTYPE", "bf16")]
GUARD_URL = os.environ.get("GUARD_URL", "http://127.0.0.1:8792/vram")
MIN_FREE_GIB = float(os.environ.get("MIN_FREE_GIB", "1.5"))     # headroom needed to (stay) on the GPU
IDLE_OFFLOAD_S = float(os.environ.get("IDLE_OFFLOAD_S", "300"))
OOM_COOLDOWN_S = float(os.environ.get("OOM_COOLDOWN_S", "120"))
FORCE_OOM_ONCE = os.environ.get("FORCE_OOM_ONCE") == "1"        # test hook for the fallback path
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
DET_MODEL = os.environ.get("DET_MODEL", "IDEA-Research/grounding-dino-tiny")
DET_TEXT = os.environ.get("DET_TEXT", "touch screen. display screen. monitor.")
# the screen is a big object: 384px short side was as accurate as the default 800 and 2.5x faster;
# bf16 only saved ~10% on the Arc and occasionally moved the box, so the detector stays fp32
DET_SIZE = [int(v) for v in os.environ.get("DET_SIZE", "384,640").split(",")]

cpu_model = SamModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
proc = SamProcessor.from_pretrained(MODEL)
lock = threading.Lock()
S = {"gpu_model": None, "det_cpu": None, "det_proc": None, "det_gpu": None, "last_used": time.time(), "cooldown_until": 0.0, "last_note": "", "fallbacks": 0}
_emb = OrderedDict()  # (image key, device) -> embeddings; the image encoder is ~all of the cost


def guard_free_gib():
    try:
        with urllib.request.urlopen(GUARD_URL, timeout=1.5) as r:
            return float(json.load(r)["free_gib"])
    except Exception:
        return None  # guard down: fail open, the OOM fallback still protects


def gpu_sync_free():
    if GPU == "xpu":
        torch.xpu.synchronize(); torch.xpu.empty_cache()


def drop_gpu(note):
    if S["gpu_model"] is not None or S["det_gpu"] is not None:
        S["gpu_model"] = S["det_gpu"] = None
        for k in [k for k in _emb if k[1] != "cpu"]:
            _emb.pop(k)
        gpu_sync_free()
        S["last_note"] = note
        print(f"[sam] GPU copy dropped: {note}", flush=True)


def pick_device():
    """Return (model, device, dtype, note)."""
    if GPU == "cpu":
        return cpu_model, "cpu", torch.float32, "gpu disabled"
    if time.time() < S["cooldown_until"]:
        return cpu_model, "cpu", torch.float32, "cpu: cooling down after GPU out-of-memory"
    free = guard_free_gib()
    if free is not None and free < MIN_FREE_GIB:
        drop_gpu(f"low VRAM ({free:.1f} GiB free)")
        return cpu_model, "cpu", torch.float32, f"cpu: only {free:.1f} GiB VRAM free"
    if S["gpu_model"] is None:
        t = time.time()
        S["gpu_model"] = copy.deepcopy(cpu_model).to(GPU, GPU_DTYPE)
        gpu_sync_free()
        print(f"[sam] GPU copy loaded in {time.time() - t:.1f}s (free before: {free})", flush=True)
    return S["gpu_model"], GPU, GPU_DTYPE, ""


def order(p):
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def mask_to_quad(mask):
    """Largest blob -> 4-corner polygon (approxPolyDP with growing epsilon, minAreaRect fallback)."""
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order(ap.reshape(4, 2).astype(np.float32))
    return order(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32))


def run_sam(model, device, dtype, img_rgb, key, box, pos, neg):
    H, W = img_rgb.shape[:2]
    bx = [box[0] * W, box[1] * H, box[2] * W, box[3] * H]
    pts = [[x * W, y * H] for x, y in pos] + [[x * W, y * H] for x, y in neg]
    pil = Image.fromarray(img_rgb)
    if pts:
        inputs = proc(pil, input_boxes=[[bx]], input_points=[[pts]],
                      input_labels=[[[1] * len(pos) + [0] * len(neg)]], return_tensors="pt")
    else:
        inputs = proc(pil, input_boxes=[[bx]], return_tensors="pt")
    pv = inputs.pop("pixel_values")
    ek = (key, device)
    cached = ek in _emb
    with torch.no_grad():
        if not cached:
            _emb[ek] = model.get_image_embeddings(pv.to(device, dtype))
            while len(_emb) > 8:
                _emb.popitem(last=False)
        emb = _emb[ek]
        prompts = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
                   for k, v in inputs.items() if k.startswith("input_")}
        out = model(image_embeddings=emb, multimask_output=True, **prompts)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.float().cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0]  # (3, H, W)
    scores = out.iou_scores[0, 0].float().cpu()
    box_area = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
    best, best_key = None, None
    for i in range(masks.shape[0]):
        m = masks[i].numpy()
        if pos:
            # must cover the inside points and avoid the outside ones; then the smallest such mask
            hit = lambda x, y: bool(m[min(H - 1, int(y * H)), min(W - 1, int(x * W))])
            cov = sum(hit(x, y) for x, y in pos) / len(pos)
            leak = sum(hit(x, y) for x, y in neg) / max(1, len(neg))
            k = (round(cov - leak, 1), -m.mean(), float(scores[i]))
        else:
            fill = m[int(bx[1]):int(bx[3]), int(bx[0]):int(bx[2])].sum() / box_area
            k = (float(scores[i]) * min(1.0, fill / 0.6),)
        if best_key is None or k > best_key:
            best, best_key = i, k
    m = masks[best].numpy()
    q = mask_to_quad(m)
    return (None if q is None else (q / [W, H]).round(4).tolist()), float(scores[best]), float(m.mean()), cached


def is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "out_of_device_memory" in s or "ur_result_error_out_of" in s \
        or isinstance(e, getattr(torch, "OutOfMemoryError", ()))


def screen(img_rgb, key, box, pos, neg):
    global FORCE_OOM_ONCE
    model, device, dtype, note = pick_device()
    try:
        if device != "cpu" and FORCE_OOM_ONCE:
            FORCE_OOM_ONCE = False
            raise RuntimeError("XPU out of memory (simulated by FORCE_OOM_ONCE)")
        res = run_sam(model, device, dtype, img_rgb, key, box, pos, neg)
    except Exception as e:
        if device == "cpu" or not is_oom(e):
            raise
        S["cooldown_until"] = time.time() + OOM_COOLDOWN_S
        S["fallbacks"] += 1
        drop_gpu(f"out of memory: {str(e)[:80]}")
        device, note = "cpu", "cpu: GPU ran out of memory, retried on CPU"
        res = run_sam(cpu_model, "cpu", torch.float32, img_rgb, key, box, pos, neg)
    if device != "cpu":
        gpu_sync_free()  # give activation memory back right away
    S["last_used"] = time.time()
    return (*res, device, note)


def det_device():
    """Same policy as pick_device(), for the detector (fp32 master on the CPU, loaded on first use)."""
    if S["det_cpu"] is None:
        t = time.time()
        S["det_proc"] = AutoProcessor.from_pretrained(DET_MODEL)
        S["det_cpu"] = AutoModelForZeroShotObjectDetection.from_pretrained(DET_MODEL).eval()
        print(f"[det] {DET_MODEL} loaded in {time.time() - t:.1f}s", flush=True)
    if GPU == "cpu" or time.time() < S["cooldown_until"]:
        return S["det_cpu"], "cpu", "gpu disabled or cooling down"
    free = guard_free_gib()
    if free is not None and free < MIN_FREE_GIB:
        drop_gpu(f"low VRAM ({free:.1f} GiB free)")
        return S["det_cpu"], "cpu", f"cpu: only {free:.1f} GiB VRAM free"
    if S["det_gpu"] is None:
        S["det_gpu"] = copy.deepcopy(S["det_cpu"]).to(GPU)
        gpu_sync_free()
    return S["det_gpu"], GPU, ""


def run_det(model, device, img_rgb):
    proc_d = S["det_proc"]
    H, W = img_rgb.shape[:2]
    inp = proc_d(images=Image.fromarray(img_rgb), text=DET_TEXT, return_tensors="pt",
                 size={"shortest_edge": DET_SIZE[0], "longest_edge": DET_SIZE[1]}).to(device)
    with torch.no_grad():
        o = model(**inp)
    r = proc_d.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.05, text_threshold=0.05,
                                                      target_sizes=[(H, W)])[0]
    if len(r["scores"]) == 0:
        return []
    clip = lambda v: min(1.0, max(0.0, v))
    out = []
    for k in r["scores"].argsort(descending=True)[:3].tolist():  # best first; rectify retries with the runner-up
        b = [float(v) for v in r["boxes"][k].cpu()]
        out.append(([clip(b[0] / W), clip(b[1] / H), clip(b[2] / W), clip(b[3] / H)], float(r["scores"][k])))
    return out


def detect(img_rgb):
    model, device, note = det_device()
    try:
        boxes = run_det(model, device, img_rgb)
    except Exception as e:
        if device == "cpu" or not is_oom(e):
            raise
        S["cooldown_until"] = time.time() + OOM_COOLDOWN_S
        S["fallbacks"] += 1
        drop_gpu(f"out of memory (detector): {str(e)[:80]}")
        device, note = "cpu", "cpu: GPU ran out of memory, retried on CPU"
        boxes = run_det(S["det_cpu"], "cpu", img_rgb)
    if device != "cpu":
        gpu_sync_free()
    S["last_used"] = time.time()
    return boxes, device, note


def idle_offloader():
    while True:
        time.sleep(15)
        with lock:
            if (S["gpu_model"] is not None or S["det_gpu"] is not None) and time.time() - S["last_used"] > IDLE_OFFLOAD_S:
                drop_gpu(f"idle for {IDLE_OFFLOAD_S:.0f}s")


class Hd(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"ok": True, "model": MODEL, "gpu": GPU, "gpu_dtype": str(GPU_DTYPE),
                         "on_gpu": S["gpu_model"] is not None, "det_model": DET_MODEL, "det_on_gpu": S["det_gpu"] is not None, "vram_free_gib": guard_free_gib(),
                         "min_free_gib": MIN_FREE_GIB, "idle_offload_s": IDLE_OFFLOAD_S,
                         "cooldown_left_s": max(0, round(S["cooldown_until"] - time.time())),
                         "fallbacks": S["fallbacks"], "last_note": S["last_note"]})

    def do_POST(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._send(400, {"error": "bad image"})
        if urlparse(self.path).path == "/detect":
            try:
                t = time.time()
                with lock:
                    boxes, device, note = detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                return self._send(200, {"box": boxes[0][0] if boxes else None, "score": round(boxes[0][1], 3) if boxes else 0.0,
                                        "boxes": [{"box": b, "score": round(sc, 3)} for b, sc in boxes],
                                        "ms": round((time.time() - t) * 1000), "device": device, "note": note})
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        try:
            box = [float(v) for v in q.get("box", "0.05,0.05,0.95,0.95").split(",")]
            parse = lambda s: [tuple(float(v) for v in p.split(",")) for p in s.split(";") if p] if s else []
            t = time.time()
            with lock:
                quad, score, area, cached, device, note = screen(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), hashlib.sha1(raw).hexdigest(), box,
                    parse(q.get("pos", "")), parse(q.get("neg", "")))
            self._send(200, {"quad": quad, "score": round(score, 3), "area": round(area, 3),
                             "ms": round((time.time() - t) * 1000), "device": device,
                             "embed_cached": cached, "note": note})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=idle_offloader, daemon=True).start()
    port = int(os.environ.get("PORT", 8791))
    print(f"sam server ({MODEL}, gpu={GPU} {GPU_DTYPE}, min free {MIN_FREE_GIB} GiB, "
          f"idle offload {IDLE_OFFLOAD_S:.0f}s) on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Hd).serve_forever()
