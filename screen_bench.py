"""Screen-crop benchmark: how well does each method find the kiosk display in a photo?

  python screen_bench.py [det_sam]          dev split (tuning photos)
  python screen_bench.py --test [det_sam]   held-out test split

Real photos only (the synthetic scenes were removed on 2026-11-18); ground truth = hand-marked corners
(fractions of width/height). Scores: polygon IoU and boundary distances (hd95 / cut95 / over95).
"""
import base64, json, os, re, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(LAB, "results")


# hand-marked display corners (tl, tr, br, bl) as fractions; shared with the test page (data/real/gt.json)
REAL_GT = {p: [tuple(c) for c in q] for p, q in json.load(open(os.path.join(LAB, "data", "real", "gt.json"))).items()}


META = json.load(open(os.path.join(LAB, "data", "real", "meta.json")))


def split_of(path):
    return META.get(path[len("real/"):], {}).get("split", "dev")


def cases(split="dev"):
    """dev = the photos the pipeline was tuned on; test = held-out photos (data/real meta split=test).
    Don't tune constants on the test split: run it only to check a change made on dev."""
    return [(p, q) for p, q in REAL_GT.items() if split_of(p) == split]


ASPECT = {}  # true aspect ratios (only the removed synthetic views had them)


def iou(q1, q2, n=500):
    a = np.zeros((n, n), np.uint8); b = np.zeros((n, n), np.uint8)
    cv2.fillPoly(a, [np.int32([[x * n, y * n] for x, y in q1])], 1)
    cv2.fillPoly(b, [np.int32([[x * n, y * n] for x, y in q2])], 1)
    inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def _dense(q, n):
    """n points per side along the closed polygon q (pixels)."""
    q = np.asarray(q, float)
    return np.concatenate([q[i] + np.linspace(0, 1, n, endpoint=False)[:, None] * (q[(i + 1) % len(q)] - q[i]) for i in range(len(q))])


def _dist_to_boundary(pts, q):
    q = np.asarray(q, float); best = np.full(len(pts), np.inf)
    for i in range(len(q)):
        a, b = q[i], q[(i + 1) % len(q)]; ab = b - a
        t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    return best


def boundary_metrics(q_pred, q_gt, W, H, n=200):
    """Boundary distances in % of the true screen's diagonal (pixel space, so x and y weigh the same).
    hd95: symmetric 95th-percentile Hausdorff distance.
    cut95: 95th percentile over the true border's points that lie OUTSIDE the prediction (screen cut off); 0 = nothing cut.
    over95: same for the prediction's border points outside the true screen (bezel/body included)."""
    P = np.array([[x * W, y * H] for x, y in q_pred]); G = np.array([[x * W, y * H] for x, y in q_gt])
    diag = max(np.linalg.norm(G[0] - G[2]), np.linalg.norm(G[1] - G[3]))
    gp, pp = _dense(G, n), _dense(P, n)
    dg, dp = _dist_to_boundary(gp, P), _dist_to_boundary(pp, G)
    inside = lambda pts, poly: np.array([cv2.pointPolygonTest(np.float32(poly), (float(x), float(y)), False) >= 0 for x, y in pts])
    cut = np.where(~inside(gp, P), dg, 0.0)
    over = np.where(~inside(pp, G), dp, 0.0)
    pct = lambda v: round(float(np.percentile(v, 95)) / diag * 100, 2)
    return {"hd95": max(pct(dg), pct(dp)), "cut95": pct(cut), "over95": pct(over)}


# ---------------- method: returns a quad as fractions (tl,tr,br,bl), or None ----------------

def m_det_sam(img, path):
    import rectify
    jpeg = open(os.path.join(LAB, "data", path), "rb").read()
    d = rectify.detect(img, jpeg)
    return [tuple(p) for p in d["quad"]] if d.get("quad") else None


METHODS = {"det_sam": m_det_sam}


def run(names, split="dev"):
    os.makedirs(RES, exist_ok=True)
    rows = []
    for path, gt in cases(split):
        img = cv2.imread(os.path.join(LAB, "data", path))
        for m in names:
            t = time.time()
            try:
                q = METHODS[m](img, path)
            except Exception as e:
                print("ERR", m, path, e); q = None
            dt = time.time() - t
            full = [(0, 0), (1, 0), (1, 1), (0, 1)]
            score = iou(q or full, gt)
            row = {"method": m, "split": split, "path": path, "iou": round(score, 3), "found": q is not None, "sec": round(dt, 2), "quad": q}
            row.update(boundary_metrics(q or full, gt, img.shape[1], img.shape[0]))
            extra = ""
            if path in ASPECT and q:
                import rectify
                h, w = img.shape[:2]
                r, src = rectify.aspect_ratio([(x * w, y * h) for x, y in q], w, h)
                tl, tr, br, bl = [np.array([x * w, y * h]) for x, y in q]
                edge = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
                est = r or edge
                row.update(aspect_true=ASPECT[path], aspect_est=round(est, 3), aspect_edge=round(float(edge), 3),
                           aspect_err=round(abs(est / ASPECT[path] - 1), 3), edge_err=round(abs(edge / ASPECT[path] - 1), 3))
                extra = f"  aspect {est:.3f} vs true {ASPECT[path]:.3f} ({src}; edges {edge:.3f})"
            rows.append(row)
            print(f"{m:10}{path:32}{score:6.2f} {'found' if q else 'none '} {dt:5.1f}s{extra}")
    with open(os.path.join(RES, "screen.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    groups = ({"close-up (own photos)": lambda p: p.startswith("real/mgc"),
               "Commons photos": lambda p: p.startswith("real/commons")} if split == "dev" else
              {"test (held-out)": lambda p: True})
    for m in names:
        rs = [r for r in rows if r["method"] == m]
        print(f"\n{m}: all {len(rs)}  HD95 {np.mean([r['hd95'] for r in rs]):.2f}  cut>2% {sum(r['cut95'] > 2 for r in rs)}"
              f"  median {np.median([r['sec'] for r in rs]):.2f}s")
        for gname, f in groups.items():
            x = [r for r in rs if f(r["path"])]
            if x:
                print(f"   {gname:22} n{len(x):2}  IoU {np.mean([r['iou'] for r in x]):.2f}  HD95 {np.mean([r['hd95'] for r in x]):5.1f}"
                      f"  cut {np.mean([r['cut95'] for r in x]):4.1f}  over {np.mean([r['over95'] for r in x]):4.1f}  cut>2% {sum(r['cut95'] > 2 for r in x)}")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--test"]
    run(args or ["det_sam"], "test" if "--test" in sys.argv else "dev")
