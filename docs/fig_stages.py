"""Collects the intermediate results that docs/gen_rectify_svg.py draws (run inside the API image, SAM server up).

  python docs/fig_stages.py <example.jpg> <failure_example.jpg> <out_dir>

Stages 1-5 are drawn on <example.jpg> (a photo that passes the checks). Stages 6-8 are drawn on
<failure_example.jpg>, a photo whose first result fails the checks, so stage 6 really runs on it.
Writes <out_dir>/stages.json, flat_check.jpg, frontal.jpg, warped.jpg.
"""
import json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import crop, rectify  # noqa: E402

ex, ex_fail, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
img = cv2.imread(ex)
H, W = img.shape[:2]
jpeg = open(ex, "rb").read()
st = {}

# 1-4: detector box, SAM candidates, consensus, bezel snapping
rough = crop.det_quads(jpeg)[0]
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

# 5: checks on the snapped quad; the flattened view with its long lines (green = within 2 deg of axis)
q = st["snap"]
st["rect_err"] = round(rectify.rect_error(q, W, H), 3)
tilt, _ = rectify.rectified_tilt(img, q)
st["tilt"] = round(tilt, 2) if tilt is not None else None
flat, _ = rectify.warp(img, q, aspect=True, long_side=800)
g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
segs = cv2.createLineSegmentDetector().detect(g)[0]
vis = (flat * 0.55 + 255 * 0.45).astype(np.uint8)
if segs is not None:
    for x1, y1, x2, y2 in segs[:, 0]:
        if np.hypot(x2 - x1, y2 - y1) < 0.08 * min(g.shape):
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        dev = min(a, abs(a - 90), 180 - a)
        if dev < 25:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (40, 160, 60) if dev < 2 else (40, 60, 220), 3)
cv2.imwrite(os.path.join(out, "flat_check.jpg"), vis)

# 6-8 on the failure example: the first detector box's result, what stage 6 did, the final result
img = cv2.imread(ex_fail)
H, W = img.shape[:2]
jpeg = open(ex_fail, "rb").read()
boxes = crop.det_quads(jpeg)
a1 = rectify._det_attempt(img, jpeg, boxes[0], W, H)
st["fail_first"] = {"quad": a1["quad"], "reasons": a1["reasons"], "rect_err": round(a1["rect_err"], 3),
                    "tilt": None if a1["tilt"] is None else round(a1["tilt"], 2)}
d = rectify.detect(img, jpeg)
st["fail_final"] = d["quad"]
st["fail_validation"] = d["validation"]
st["fail_tried"] = d["tried"]
kept = next((t for t in d["tried"] if t.startswith("validation:")), "")
k = int(kept.rsplit("kept box ", 1)[1]) - 1 if "kept box " in kept else 0
before = rectify._det_attempt(img, jpeg, boxes[k], W, H)["quad"]  # chosen attempt before stage 7
st["fail_kept_box"] = k + 1
st["fail_before_extent"] = before
qs = st["fail_first"]["quad"] + st["fail_final"]
xs, ys = [x for x, _ in qs], [y for _, y in qs]
st["fail_view"] = [max(0, min(xs) - 0.12), max(0, min(ys) - 0.06), min(1, max(xs) + 0.12), min(1, max(ys) + 0.06)]

# 7: frontal view with an 8% margin, before and after re-deciding the sides
fq = np.float32([[x * W, y * H] for x, y in st["fail_final"]])
r, _ = rectify.aspect_ratio([tuple(p) for p in fq], W, H)
r = r or 1.33
w, h = (800, 800 / r) if r >= 1 else (800 * r, 800)
m = 0.08
M = cv2.getPerspectiveTransform(fq, np.float32([[m * w, m * h], [m * w + w, m * h], [m * w + w, m * h + h], [m * w, m * h + h]]))
front = cv2.warpPerspective(img, M, (int(w * (1 + 2 * m)), int(h * (1 + 2 * m))))
sq = cv2.perspectiveTransform(np.float32([[[x * W, y * H] for x, y in before]]), M)[0]
st["frontal_before"] = [[float(x), float(y)] for x, y in sq]
st["frontal_after"] = [[m * w, m * h], [m * w + w, m * h], [m * w + w, m * h + h], [m * w, m * h + h]]
st["extent_note"] = next((t for t in d["tried"] if t.startswith("frontal extent")), "")
cv2.imwrite(os.path.join(out, "frontal.jpg"), front)

# 8: flatten
warped, info = rectify.warp(img, st["fail_final"], aspect=True, long_side=1600)
st["aspect"] = info
cv2.imwrite(os.path.join(out, "warped.jpg"), warped)

json.dump(st, open(os.path.join(out, "stages.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: st[k] for k in ("box_share", "snap_sides", "rect_err", "tilt", "fail_first", "fail_validation",
                                     "fail_kept_box", "extent_note", "aspect")}, ensure_ascii=False))
