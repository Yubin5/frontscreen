"""Collects the intermediate results that docs/gen_rectify_svg.py draws (run inside the API/lab image, SAM server up).

  python docs/fig_stages.py <example.jpg> <retry_example.jpg> <out_dir>

Writes <out_dir>/stages.json, warped.jpg, flat_check.jpg, frontal.jpg.
"""
import json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import crop, rectify, screen_ocr  # noqa: E402

ex, ex_retry, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
img = cv2.imread(ex)
H, W = img.shape[:2]
jpeg = open(ex, "rb").read()
st = {}

# 1-4: detector box, SAM candidates, consensus, bezel snapping
boxes = crop.det_quads(jpeg)
rough = boxes[0]
st["rough"] = [list(p) for p in rough]
st["box_share"] = round(crop.area(rough), 3)
pos, neg = rectify.grid_points(rough)
st["pos"], st["neg"] = [list(p) for p in pos], [list(p) for p in neg]
cands = [c for c in rectify.sam_candidates(jpeg, rough) if "quad" in c and crop.iou(c["quad"], rough) >= 0.3]
st["cands"] = [c["quad"] for c in cands]
cons = rectify.consensus(st["cands"], W, H)
st["consensus"] = [list(p) for p in cons]
snap, moved = rectify.bezel_snap_outer(img, cons)
st["snap"] = [list(p) for p in snap] if any(moved) else st["consensus"]
st["snap_sides"] = int(sum(moved))

# 5: checks on the snapped quad
q = st["snap"]
st["rect_err"] = round(rectify.rect_error(q, W, H), 3)
tilt, nlines = rectify.rectified_tilt(img, q)
st["tilt"] = round(tilt, 2) if tilt is not None else None
st["tilt_lines"] = nlines
flat, _ = rectify.warp(img, q, aspect=True, long_side=800)
g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
lsd = cv2.createLineSegmentDetector()
segs = lsd.detect(g)[0]
vis = (flat * 0.55 + 255 * 0.45).astype(np.uint8)
if segs is not None:
    for x1, y1, x2, y2 in segs[:, 0]:
        L = np.hypot(x2 - x1, y2 - y1)
        if L < 0.08 * min(g.shape):
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        dev = min(a, abs(a - 90), 180 - a)
        if dev < 25:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (40, 160, 60) if dev < 2 else (40, 60, 220), 3)
cv2.imwrite(os.path.join(out, "flat_check.jpg"), vis)

# 7: frontal view with an 8% margin, before and after re-deciding the sides
d = rectify.detect(img, jpeg)
st["final"] = d["quad"]
st["validation"] = d["validation"]
st["tried"] = d["tried"]
fq = np.float32([[x * W, y * H] for x, y in st["final"]])
r, _ = rectify.aspect_ratio([tuple(p) for p in fq], W, H)
r = r or 1.33
w, h = (800, 800 / r) if r >= 1 else (800 * r, 800)
m = 0.08
M = cv2.getPerspectiveTransform(fq, np.float32([[m * w, m * h], [m * w + w, m * h], [m * w + w, m * h + h], [m * w, m * h + h]]))
front = cv2.warpPerspective(img, M, (int(w * (1 + 2 * m)), int(h * (1 + 2 * m))))
sq = cv2.perspectiveTransform(np.float32([[[x * W, y * H] for x, y in st["snap"]]]), M)[0]
st["frontal_size"] = [front.shape[1], front.shape[0]]
st["frontal_before"] = [[float(x), float(y)] for x, y in sq]
st["frontal_after"] = [[m * w, m * h], [m * w + w, m * h], [m * w + w, m * h + h], [m * w, m * h + h]]
cv2.imwrite(os.path.join(out, "frontal.jpg"), front)

# 8-9: flatten + OCR
warped, info = rectify.warp(img, st["final"], aspect=True, long_side=1600)
st["aspect"] = info
cv2.imwrite(os.path.join(out, "warped.jpg"), warped)
o = screen_ocr.read(warped)
st["warped_size"] = [warped.shape[1], warped.shape[0]]
st["ocr"] = o["lines"]
st["ocr_ms"] = o["ms"]

# 6: retry example (first box fails the checks, another box passes better)
img2 = cv2.imread(ex_retry)
H2, W2 = img2.shape[:2]
jpeg2 = open(ex_retry, "rb").read()
att = []
for b in crop.det_quads(jpeg2)[:3]:
    a = rectify._det_attempt(img2, jpeg2, b, W2, H2)
    att.append({"box": [list(p) for p in b], "quad": a["quad"], "rect_err": round(a["rect_err"], 3),
                "tilt": None if a["tilt"] is None else round(a["tilt"], 2), "severity": round(rectify._severity(a), 2)})
st["retry"] = att
json.dump(st, open(os.path.join(out, "stages.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: st[k] for k in ("box_share", "snap_sides", "rect_err", "tilt", "validation", "aspect")}, ensure_ascii=False))
print("retry severities", [a["severity"] for a in att], "| ocr lines", len(st["ocr"]), st["ocr_ms"], "ms")
print("\n".join(st["tried"][:3]))
