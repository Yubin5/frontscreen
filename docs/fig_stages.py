"""Collects the intermediate results that docs/gen_rectify_svg.py draws (run inside the API image, SAM server up).

  python docs/fig_stages.py <example.jpg> <out_dir>

Writes <out_dir>/stages.json, warped.jpg, flat_check.jpg, frontal.jpg.
Stage 6 (second pass) does not run on a photo that passes the checks, so it is shown on a copy of the final
quad with one corner moved by SKEW (labelled as such in the figure).
"""
import json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import crop, rectify  # noqa: E402

SKEW = (1, (0.0, 0.10))  # corner index (tl, tr, br, bl) and its shift in fractions of width/height

ex, out = sys.argv[1:3]
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

# full pipeline result
d = rectify.detect(img, jpeg)
st["final"] = d["quad"]
st["validation"] = d["validation"]
st["tried"] = d["tried"]

# 6: second pass on a deliberately skewed copy of the final quad
k, (dx, dy) = SKEW
sk = [list(p) for p in st["final"]]
sk[k] = [sk[k][0] + dx, sk[k][1] + dy]
st["skewed"] = sk
st["skewed_rect_err"] = round(rectify.rect_error(sk, W, H), 3)
st["skewed_tilt"] = round(rectify.rectified_tilt(img, sk)[0], 2)
fixed, note = rectify.second_pass(img, sk)
st["fixed"], st["fixed_note"] = fixed, note
st["fixed_tilt"] = round(rectify.rectified_tilt(img, fixed)[0], 2) if fixed else None

# 7: frontal view with an 8% margin, before (bezel-snapped) and after re-deciding the sides
fq = np.float32([[x * W, y * H] for x, y in st["final"]])
r, _ = rectify.aspect_ratio([tuple(p) for p in fq], W, H)
r = r or 1.33
w, h = (800, 800 / r) if r >= 1 else (800 * r, 800)
m = 0.08
M = cv2.getPerspectiveTransform(fq, np.float32([[m * w, m * h], [m * w + w, m * h], [m * w + w, m * h + h], [m * w, m * h + h]]))
front = cv2.warpPerspective(img, M, (int(w * (1 + 2 * m)), int(h * (1 + 2 * m))))
sq = cv2.perspectiveTransform(np.float32([[[x * W, y * H] for x, y in st["snap"]]]), M)[0]
st["frontal_before"] = [[float(x), float(y)] for x, y in sq]
st["frontal_after"] = [[m * w, m * h], [m * w + w, m * h], [m * w + w, m * h + h], [m * w, m * h + h]]
cv2.imwrite(os.path.join(out, "frontal.jpg"), front)

# 8: flatten
warped, info = rectify.warp(img, st["final"], aspect=True, long_side=1600)
st["aspect"] = info
cv2.imwrite(os.path.join(out, "warped.jpg"), warped)

json.dump(st, open(os.path.join(out, "stages.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: st[k] for k in ("box_share", "snap_sides", "rect_err", "tilt", "validation", "aspect",
                                     "skewed_rect_err", "skewed_tilt", "fixed_tilt", "fixed_note")}, ensure_ascii=False))
