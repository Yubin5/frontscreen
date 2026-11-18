"""Helpers for the screen finder (see rectify.py): detector boxes and SAM calls to the GPU server, polygon IoU /
area, and crop() for the phone view (detect + flatten).
"""
import base64, json, os, re, time, urllib.request

import cv2
import numpy as np


def _post(url, data, ctype, timeout=600):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

SAM_URL = os.environ.get("SAM_URL", "http://127.0.0.1:8791/screen")


DET_URL = os.environ.get("DET_URL", "http://127.0.0.1:8791/detect")


def det_quads(jpeg):
    """Screen boxes from the open-vocabulary detector (Grounding DINO tiny in the SAM container), best first,
    as quads (tl,tr,br,bl); tiny boxes dropped."""
    r = _post(DET_URL, jpeg, "application/octet-stream")
    out = []
    for e in r.get("boxes") or ([{"box": r["box"]}] if r.get("box") else []):
        b = e["box"]
        q = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
        if area(q) > 0.01:
            out.append(q)
    return out


def det_quad(jpeg):
    """Best detector box as a quad, or None."""
    qs = det_quads(jpeg)
    return qs[0] if qs else None


def iou(q1, q2, n=400):
    a = np.zeros((n, n), np.uint8); b = np.zeros((n, n), np.uint8)
    cv2.fillPoly(a, [np.int32([[x * n, y * n] for x, y in q1])], 1)
    cv2.fillPoly(b, [np.int32([[x * n, y * n] for x, y in q2])], 1)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def area(q):
    return float(cv2.contourArea(np.float32(q)))


def sam(jpeg, box, pos=(), neg=()):
    fmt = lambda pts: ";".join(f"{x:.4f},{y:.4f}" for x, y in pts)
    url = f"{SAM_URL}?box=" + ",".join(f"{v:.4f}" for v in box)
    if pos:
        url += f"&pos={fmt(pos)}&neg={fmt(neg)}"
    r = _post(url, jpeg, "application/octet-stream")
    sam.last = r  # score / mask area for callers that rank several prompts
    return [tuple(p) for p in r["quad"]] if r.get("quad") else None, r.get("ms", 0)


def warp(img, quad, margin=0.01):
    H, W = img.shape[:2]
    q = np.float32([[x * W, y * H] for x, y in quad])
    c = q.mean(0); q = (q - c) * (1 + margin * 2) + c  # keep a hair of bezel so edge buttons aren't clipped
    tl, tr, br, bl = q
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if max(out.shape[:2]) < 1400:  # far-away screens: enlarge for OCR and display
        f = min(3.0, 1400 / max(out.shape[:2]))
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return out


def crop(img, jpeg, run_ocr=None):
    """Phone-view crop: image-only detection + perspective/aspect correction (see rectify.py).
    run_ocr is no longer used; kept so callers don't change."""
    import rectify
    t0 = time.time()
    d = rectify.detect(img, jpeg)
    ms = d.get("ms", {})
    if not d.get("quad"):
        return {"cropped": False, "reason": "화면을 찾지 못함", "ms": ms}
    if area(d["quad"]) > 0.85:
        return {"cropped": False, "reason": "화면이 이미 사진을 거의 채움", "rough": d.get("rough"), "ms": ms}
    out, info = rectify.warp(img, d["quad"], aspect=True)
    ms["total"] = round((time.time() - t0) * 1000)
    return {"cropped": True, "quad": d["quad"], "rough": d.get("rough"), "used": d["used"], "tried": d.get("tried", []),
            "w": out.shape[1], "h": out.shape[0], "ms": ms, **info,
            "image": base64.b64encode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()}
