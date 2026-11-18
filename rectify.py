"""Image-only screen rectification (no OCR): find the display, then flatten it to a front view.
Detector + SAM only (the VLM and contour methods were removed on 2026-11-18).

detect(img, jpeg) -> {quad, rough, used, validation, tried, ms}
    1. Grounding DINO tiny box (crop.det_quads, ~0.2s)
    2. SAM candidates from that box: box with inside/outside grid points, and box only, at 2-7% padding
    3. per-corner median of the candidates (consensus)
    4. close-ups (box 35-80% of the photo): sides moved to the outermost border inside the bezel
       (bezel_snap_outer; BEZEL_MODE = "inner" uses bezel_snap)
    5. checks: 3D right angles (rect_error), straight lines inside the flattened screen (rectified_tilt),
       bezel border found on >= 2 sides; on a failed shape check the perspective is re-estimated from the
       screen's own lines (second_pass)
    6. close-ups: final per-side extent from row/column profiles of the frontal view (frontal_extent)
warp(img, quad, aspect=True) -> front-facing image; with aspect=True the screen's true width/height
    ratio is estimated from the perspective (Zhang & He, "whiteboard scanning"), so oblique shots
    don't come out squashed.
"""
import base64, time

import cv2
import numpy as np

import crop  # det_quads, sam, iou, area


def _bilinear(q, u, v):
    tl, tr, br, bl = [np.array(p) for p in q]
    return tuple((tl * (1 - u) + tr * u) * (1 - v) + (bl * (1 - u) + br * u) * v)


def grid_points(q):
    pos = [_bilinear(q, u, v) for u, v in ((0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))]
    c = np.mean(np.array(q), 0)
    ring = [tuple(c + (np.array(p) - c) * 1.35) for p in
            list(q) + [_bilinear(q, 0.5, 0), _bilinear(q, 1, 0.5), _bilinear(q, 0.5, 1), _bilinear(q, 0, 0.5)]]
    neg = [p for p in ring if 0.01 < p[0] < 0.99 and 0.01 < p[1] < 0.99]
    return pos, neg


def candidates(img, jpeg, method="det_sam"):
    """Detector box + every SAM candidate (for offline work, see det_cands.py). -> (rough, cands, ms)"""
    ms = {}
    t = time.time(); rough = crop.det_quad(jpeg); ms["det"] = round((time.time() - t) * 1000)
    if not rough:
        return rough, [], ms
    t = time.time(); cands = sam_candidates(jpeg, rough, grid=True); ms["sam"] = round((time.time() - t) * 1000)
    return rough, cands, ms


def sam_candidates(jpeg, rough, grid=True):
    """Every SAM candidate for a rough quad: box (+2..7% pad) with inside/outside grid points, and box only."""
    xs, ys = [p[0] for p in rough], [p[1] for p in rough]
    boxf = lambda pad: [max(0, min(xs) - pad), max(0, min(ys) - pad), min(1, max(xs) + pad), min(1, max(ys) + pad)]
    pos, neg = grid_points(rough)
    # SAM is sensitive to small prompt changes, so try several (cheap once the image embedding is cached)
    prompts = [("sam_grid", 0.04, pos, neg), ("sam_grid", 0.07, pos, neg)] if grid else []
    prompts += [("sam_box", 0.02, (), ()), ("sam_box", 0.04, (), ()), ("sam_box", 0.07, (), ())]
    cands = []
    for name, pad, p, n in prompts:
        try:
            q, _ = crop.sam(jpeg, boxf(pad), p, n)
        except Exception as e:
            cands.append({"name": name, "pad": pad, "error": str(e)}); continue
        if q:
            r = crop.sam.last
            cands.append({"name": name, "pad": pad, "quad": [list(x) for x in q], "score": r.get("score", 0), "mask": r.get("area", 0)})
    return cands


def consensus(quads, W, H, qtl=0.5):
    """Per corner, the qtl-quantile (0.5 = median) of the candidates' corners along the centroid->corner direction,
    as a continuous value (picking one candidate's corner made the result jump when a candidate shifted).
    SAM candidates that cut the screen sit inside the others and ones that leak onto the body sit outside,
    so the middle one is rarely either; measured in pixels so x and y weigh the same."""
    Q = np.array([[[x * W, y * H] for x, y in q] for q in quads])
    cen = Q.reshape(-1, 2).mean(0)
    out = []
    for k in range(4):
        pts = Q[:, k]
        d = np.median(pts, 0) - cen
        d /= max(np.linalg.norm(d), 1e-9)
        proj = (pts - cen) @ d
        out.append(cen + d * np.quantile(proj, qtl) + (np.median(pts, 0) - cen - d * ((np.median(pts, 0) - cen) @ d)))
    return [[round(float(x / W), 4), round(float(y / H), 4)] for x, y in out]


def bezel_snap_outer(img, quad, out_frac=0.12, in_frac=0.30, n=48, min_share=0.45, max_turn=30, min_step=12, margin=0.006):
    """Outermost display border inside the bezel. Per profile (inside -> outside): the screen's bright interior
    sets a darkness threshold; the dark run near the start border is bezel (+ the dimmed screen around a modal
    popup); its outer end, where the body/background begins, bounds the search. The outermost brightness step
    of either sign strictly inside that bound is the candidate (a popup's border is always further in).
    RANSAC over profiles as in bezel_snap. Profiles without a bounded dark run (dark background) are skipped."""
    H, W = img.shape[:2]
    s = min(1.0, 900 / max(H, W))
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY),
                         (5, 5), 0).astype(np.float32)
    h, w = g.shape
    q = np.float64([[x * w, y * h] for x, y in quad])
    cen, size = q.mean(0), np.sqrt(abs(cv2.contourArea(np.float32(q))))
    offs = np.arange(-in_frac * size, out_frac * size, 1.0)  # inside -> outside
    z = int(np.argmin(np.abs(offs)))
    k = 3
    rng = np.random.default_rng(0)
    lines, moved = [], []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        nv = np.array([-t[1], t[0]])
        if (cen - p0) @ nv > 0:
            nv = -nv  # outward
        cands, used = [], 0
        for pi, u in enumerate(np.linspace(0.12, 0.88, n)):
            b = p0 + u * (p1 - p0)
            xs, ys = b[0] + offs * nv[0], b[1] + offs * nv[1]
            ok = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
            if ok[:z].sum() < 0.5 * z:
                continue
            used += 1
            prof = np.full(len(offs), np.nan)
            prof[ok] = cv2.remap(g, xs[ok].astype(np.float32)[None], ys[ok].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            bright = np.nanpercentile(prof[:z], 75)  # the lit screen (or the popup on it)
            T = 0.5 * bright
            dark = np.nan_to_num(prof, nan=255.0) < T
            # dark run reaching the start border from the inside or starting just outside it
            js = [j for j in range(max(0, z - int(0.25 * size)), min(len(offs), z + int(0.06 * size))) if dark[j]]
            if not js:
                continue
            j0 = js[0]
            j1 = j0
            while j1 + 1 < len(offs) and (dark[j1 + 1] or (j1 + 2 < len(offs) and dark[j1 + 2])):
                j1 += 1
            if j1 + 1 >= len(offs) or np.isnan(prof[min(len(offs) - 1, j1 + 3)]):
                continue  # the dark run never ends inside the window: no body/background edge to bound it
            bound = j1 - int(margin * size)  # stay inside the bezel's outer edge
            step = np.zeros(len(offs))
            step[k:-k] = np.nan_to_num(prof[2 * k:] - prof[:-2 * k])
            js2 = [j for j in range(max(k, j0 - k), bound) if abs(step[j]) >= min_step and abs(step[j]) == np.abs(step[j - k:j + k + 1]).max()]
            if js2:
                j = js2[-1]  # outermost edge of either sign inside the bezel
                cands.append((pi, b + offs[j] * nv, offs[j]))
        line = None
        if used >= n // 3 and len(cands) >= 2:
            line, _ = _support_line(cands, t, size, used, min_share, max_turn, rng)
        moved.append(line is not None)
        lines.append(line or (p0, t))
    if not any(moved):
        return quad, moved

    def cross(l1, l2):
        (a, u), (b, v) = l1, l2
        A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
        return None if abs(np.linalg.det(A)) < 1e-6 else a + np.linalg.solve(A, b - a)[0] * u

    new = [cross(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    if any(p is None for p in new):
        return quad, [False] * 4
    new = np.array(new)
    if not cv2.isContourConvex(np.float32(new)) or not 0.5 < cv2.contourArea(np.float32(new)) / cv2.contourArea(np.float32(q)) < 1.6:
        return quad, [False] * 4
    return [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in new], moved


def _support_line(cands, t, size, used, min_share, max_turn, rng, iters=300):
    """RANSAC over (profile index, point, offset): the line most profiles agree on (outermost on ties).
    -> ((point, dir), mean offset) or (None, None)"""
    if len(cands) < 2:
        return None, None
    P = np.array([c[1] for c in cands]); prof_id = np.array([c[0] for c in cands]); off = np.array([c[2] for c in cands])
    best = None
    for _ in range(iters):
        a, b = rng.choice(len(P), 2, replace=False)
        if prof_id[a] == prof_id[b]:
            continue
        d = P[b] - P[a]
        L = np.linalg.norm(d)
        if L < 1e-6 or abs(d @ t) / L < np.cos(np.radians(max_turn)):
            continue
        d /= L
        inl = np.abs((P - P[a]) @ np.array([-d[1], d[0]])) < 0.01 * size
        key = (len(set(prof_id[inl])), off[inl].mean())
        if best is None or key > best[0]:
            best = (key, inl)
    if best is None or best[0][0] < min_share * used:
        return None, None
    vx, vy, x0, y0 = cv2.fitLine(np.float32(P[best[1]]), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return (np.array([x0, y0]), np.array([vx, vy])), best[0][1]


BEZEL_SNAP_AREA = (0.35, 0.80)  # detector-box share of the photo where bezel snapping runs (close-ups with the bezel in view)
# "outer": outermost border inside the bezel (bezel_snap_outer) - never stops at a modal popup's border when the
#          bezel's outer edge is visible, at the price of a little more bezel; "inner": first bezel->screen rise
BEZEL_MODE = "outer"


def bezel_snap(img, quad, out_frac=0.08, in_frac=0.30, n=48, min_share=0.45, dark=0.6, max_turn=30):
    """Move each side onto the lit-screen border (the inner edge of the bezel).
    Along the outward normal at n points per side, every brightness rise going inward whose outer side is
    clearly darker than its inner side (bezel -> screen) is a candidate; a RANSAC line through the candidates
    of all n profiles (up to max_turn degrees off the starting side, which may sit on the body at a different
    angle) keeps the line most profiles agree on (the bezel's inner edge runs the whole side,
    a dark UI panel touching it does not), outermost on ties. Searching 30% inward lets a side that started
    on the kiosk body (or at the photo border) still reach the screen. Sides without such a line are kept.
    -> (quad, sides moved). Only reliable with the bezel in view; see BEZEL_SNAP_AREA."""
    H, W = img.shape[:2]
    s = min(1.0, 900 / max(H, W))
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY),
                         (5, 5), 0).astype(np.float32)
    h, w = g.shape
    q = np.float64([[x * w, y * h] for x, y in quad])
    cen, size = q.mean(0), np.sqrt(abs(cv2.contourArea(np.float32(q))))
    offs = np.arange(out_frac * size, -in_frac * size, -1.0)  # outside -> inside
    k, band = 3, max(3, int(0.012 * size))
    rng = np.random.default_rng(0)
    lines, moved = [], []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        nv = np.array([-t[1], t[0]])
        if (cen - p0) @ nv > 0:
            nv = -nv  # outward
        cands, used = [], 0  # (profile index, point, offset)
        for pi, u in enumerate(np.linspace(0.12, 0.88, n)):
            b = p0 + u * (p1 - p0)
            xs, ys = b[0] + offs * nv[0], b[1] + offs * nv[1]
            ok = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
            if ok.sum() < 0.3 * len(offs):
                continue
            used += 1
            prof = np.full(len(offs), np.nan)
            prof[ok] = cv2.remap(g, xs[ok].astype(np.float32)[None], ys[ok].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            step = np.zeros(len(offs))
            step[k:-k] = prof[2 * k:] - prof[:-2 * k]  # brightness gain moving inward
            step = np.nan_to_num(step)
            peaks = [j for j in range(band, len(step) - band)
                     if step[j] >= 25 and step[j] == step[j - k:j + k + 1].max()]
            for j in peaks:
                o_seg, i_seg = prof[max(0, j - band - k):j - k], prof[j + k:j + k + band]
                if not (np.isfinite(o_seg).any() and np.isfinite(i_seg).any()):
                    continue
                outer, inner = np.nanmedian(o_seg), np.nanmedian(i_seg)
                if np.isfinite(outer) and np.isfinite(inner) and outer <= dark * inner:
                    cands.append((pi, b + offs[j] * nv, offs[j]))
        line = None
        if used >= n // 3 and len(cands) >= 2:
            P = np.array([c[1] for c in cands]); prof_id = np.array([c[0] for c in cands]); off = np.array([c[2] for c in cands])
            best = None
            for _ in range(300):
                a, b2 = rng.choice(len(P), 2, replace=False)
                if prof_id[a] == prof_id[b2]:
                    continue
                d = P[b2] - P[a]
                L = np.linalg.norm(d)
                if L < 1e-6 or abs(d @ t) / L < np.cos(np.radians(max_turn)):
                    continue
                d /= L
                inl = np.abs((P - P[a]) @ np.array([-d[1], d[0]])) < 0.01 * size
                support = len(set(prof_id[inl]))  # count profiles, not points
                key = (support, off[inl].mean())  # most profiles, then outermost
                if best is None or key > best[0]:
                    best = (key, inl)
            if best is not None and best[0][0] >= min_share * used:
                inl = best[1]
                vx, vy, x0, y0 = cv2.fitLine(np.float32(P[inl]), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                line = (np.array([x0, y0]), np.array([vx, vy]))
        moved.append(line is not None)
        lines.append(line or (p0, t))
    if not any(moved):
        return quad, moved

    def cross(l1, l2):
        (a, u), (b, v) = l1, l2
        A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
        return None if abs(np.linalg.det(A)) < 1e-6 else a + np.linalg.solve(A, b - a)[0] * u

    new = [cross(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    if any(p is None for p in new):
        return quad, [False] * 4
    new = np.array(new)
    if not cv2.isContourConvex(np.float32(new)) or not 0.5 < cv2.contourArea(np.float32(new)) / cv2.contourArea(np.float32(q)) < 1.4:
        return quad, [False] * 4
    return [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in new], moved


FRONTAL_EXTENT = True  # final per-side extent from row/column profiles of the frontal view
FRONTAL_AREA = (0.35, 0.80)  # detector-box share of the photo where frontal_extent runs
SECOND_PASS = True  # re-estimate the perspective from the screen's own lines when the result is not straight
SECOND_PASS_TILT = 1.5  # degrees of leftover tilt inside the flattened screen that trigger the second pass


def _vanishing_point(segs, w):
    """Least-squares common point of line segments (homogeneous), with two rounds of dropping segments that point
    more than 2 deg away from it (reflections, slanted graphics). -> (vp, kept mask)"""
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:4], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    keep = np.ones(len(segs), bool)
    vp = None
    for _ in range(3):
        if keep.sum() < 2:
            return None, keep
        A = lines[keep] * np.sqrt(w[keep])[:, None]
        vp = np.linalg.svd(A.T @ A)[2][-1]
        mid = (segs[:, :2] + segs[:, 2:4]) / 2
        d_seg = segs[:, 2:4] - segs[:, :2]
        d_vp = vp[:2][None] - mid * vp[2]  # direction from the segment midpoint towards vp (works at infinity too)
        cosang = np.abs(np.sum(d_seg * d_vp, 1)) / (np.linalg.norm(d_seg, axis=1) * np.linalg.norm(d_vp, axis=1) + 1e-9)
        keep = np.degrees(np.arccos(np.clip(cosang, 0, 1))) < 2.0
    return vp, keep


def second_pass(img, quad, margin=0.06, side=1000):
    """Straighten a skewed result with the display's own lines. Flatten a slightly enlarged quad, find the long
    near-horizontal and near-vertical lines (popup border, buttons, menu grid, text rows: all in the screen's
    plane), take their two vanishing points, and build the correction that sends both to infinity. The first
    quad's sides are then averaged into an axis-aligned rectangle in the corrected view and mapped back.
    -> (quad or None, note)"""
    global _lsd
    if _lsd is None:
        _lsd = cv2.createLineSegmentDetector()
    H_, W_ = img.shape[:2]
    q = np.float64([[x * W_, y * H_] for x, y in quad])
    c = q.mean(0)
    qe = (q - c) * (1 + 2 * margin) + c
    r, _ = aspect_ratio([tuple(p) for p in q], W_, H_)
    if not r:
        tl, tr, br, bl = q
        r = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1)
    w1, h1 = (side, int(side / r)) if r >= 1 else (int(side * r), side)
    M1 = cv2.getPerspectiveTransform(np.float32(qe), np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]))
    flat = cv2.warpPerspective(img, M1, (w1, h1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    segs = _lsd.detect(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY))[0]
    if segs is None:
        return None, "no lines"
    segs = segs[:, 0].astype(np.float64)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs, L = segs[L > 0.06 * min(w1, h1)], L[L > 0.06 * min(w1, h1)]
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    hor = np.minimum(ang, 180 - ang) < 20
    ver = np.abs(ang - 90) < 20
    if hor.sum() < 3 or ver.sum() < 3:
        return None, f"too few lines ({hor.sum()} horizontal, {ver.sum()} vertical)"
    vh, kh = _vanishing_point(segs[hor], L[hor])
    vv, kv = _vanishing_point(segs[ver], L[ver])
    if vh is None or vv is None or kh.sum() < 3 or kv.sum() < 3:
        return None, "lines do not agree on a vanishing point"
    l_inf = np.cross(vh, vv)
    if abs(l_inf[2]) < 1e-12:
        return None, "degenerate"
    l_inf /= l_inf[2]
    Hp = np.array([[1, 0, 0], [0, 1, 0], [l_inf[0], l_inf[1], 1.0]])
    dh, dv = (Hp @ vh)[:2], (Hp @ vv)[:2]
    if dh[0] < 0:
        dh = -dh
    if dv[1] < 0:
        dv = -dv
    A2 = np.linalg.inv(np.c_[dh / np.linalg.norm(dh), dv / np.linalg.norm(dv)])
    H2 = np.eye(3)
    H2[:2, :2] = A2
    H2 = H2 @ Hp
    T = H2 @ M1  # original image -> corrected frontal view
    qp = cv2.perspectiveTransform(np.float32(q)[None], T)[0]
    top, bottom = (qp[0, 1] + qp[1, 1]) / 2, (qp[2, 1] + qp[3, 1]) / 2
    left, right = (qp[0, 0] + qp[3, 0]) / 2, (qp[1, 0] + qp[2, 0]) / 2
    if not (right > left and bottom > top):
        return None, "flipped"
    rect = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    back = cv2.perspectiveTransform(rect[None], np.linalg.inv(T))[0]
    if not cv2.isContourConvex(back):
        return None, "not convex"
    out = [[round(float(x / W_), 4), round(float(y / H_), 4)] for x, y in back]
    return out, f"straightened with {int(kh.sum())} horizontal + {int(kv.sum())} vertical lines"


def frontal_extent(img, quad, margin=0.08, in_frac=0.15, side=900, k=2, edge_min=6.0, max_in=0.05, max_out=0.02, nseg=2,
                   max_in_body=0.15):
    """Re-decide where each side of the display ends, in a frontal view (the quad's perspective is trusted: it
    passed the checks or was straightened). The quad is flattened with an 8% margin (the homography extended
    past its corners, so the true border is in view). Each side is split into nseg parts; per part one
    brightness profile is taken across the border (median of each row or column over that part). Walking
    outward: the lit screen ends where a dark band (bezel, plus the dimmed screen around a popup) starts; if a
    further darkening step follows before the first clear brightening (bezel -> body/background), the outermost
    such step is the edge (dimmed screen -> bezel), otherwise the band's start is. A part moves at most max_in inward / max_out outward (bigger moves were a
    popup's border or the body on the bench). The parts' edge points give each side's line, so one corner
    that sits on the bezel can come back on its own. -> (quad, per-side note)"""
    H_, W_ = img.shape[:2]
    q = np.float64([[x * W_, y * H_] for x, y in quad])
    r, _ = aspect_ratio([tuple(p) for p in q], W_, H_)
    if not r:
        tl, tr, br, bl = q
        r = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1)
    w, h = (side, side / r) if r >= 1 else (side * r, side)
    mx, my = margin * w, margin * h
    cw, ch = int(round(w + 2 * mx)), int(round(h + 2 * my))
    M = cv2.getPerspectiveTransform(np.float32(q), np.float32([[mx, my], [mx + w, my], [mx + w, my + h], [mx, my + h]]))
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.warpPerspective(img, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)),
                                      cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    valid = cv2.warpPerspective(np.full((H_, W_), 255, np.uint8), M, (cw, ch), flags=cv2.INTER_NEAREST) > 0
    g[~valid] = np.nan
    base = {"T": my, "B": my + h, "L": mx, "R": mx + w}
    lines, notes = {}, {}
    for sd in "TRBL":
        horiz = sd in "TB"
        L_ = h if horiz else w          # depth scale for this side
        along0, along_len = (mx, w) if horiz else (my, h)
        pos = base[sd]
        span = in_frac * L_
        pts, part_notes = [], []
        for si in range(nseg):
            a0 = along0 + along_len * (0.15 + 0.70 * si / nseg)
            a1 = along0 + along_len * (0.15 + 0.70 * (si + 1) / nseg)
            with np.errstate(all="ignore"):
                prof = np.nanmedian(g[:, int(a0):int(a1)], axis=1) if horiz else np.nanmedian(g[int(a0):int(a1), :], axis=0)
            idx = np.arange(int(pos - span), len(prof)) if sd in "BR" else np.arange(int(pos + span), -1, -1)
            idx = idx[(idx >= 0) & (idx < len(prof))]
            pr = prof[idx]
            if len(pr) < 10 or np.isnan(pr[: max(3, len(pr) // 4)]).any():
                part_notes.append("off-photo"); continue
            bright = np.nanpercentile(pr[: int(span)], 75)
            dark = (np.nan_to_num(pr, nan=-1.0) < 0.5 * bright) & ~np.isnan(pr)
            if not dark.any():
                part_notes.append("no dark band"); continue
            j0 = int(np.argmax(dark))
            j1 = j0
            while j1 + 1 < len(pr) and (dark[j1 + 1] or (j1 + 2 < len(pr) and dark[j1 + 2])):
                j1 += 1
            cut = j0
            d = np.zeros(len(pr))
            seg = np.nan_to_num(pr, nan=np.nanmin(pr))
            d[k:-k] = seg[2 * k:] - seg[:-2 * k]  # brightness change going outward
            # past the screen edge: the band ends at the first clear brightening (bezel -> body/background);
            # before that, a further darkening step (dimmed screen around a popup -> bezel) is the real edge
            rises = [jj for jj in range(j0 + 2 * k, len(pr) - k) if d[jj] >= 2 * edge_min]
            bound = rises[0] if rises else len(pr) - k
            inner = [jj for jj in range(j0 + 2 * k, bound) if d[jj] <= -edge_min and d[jj] == d[max(0, jj - k):jj + k + 1].min()]
            if inner:
                cut = inner[-1]
            out = (idx[cut] - pos) * (1 if sd in "BR" else -1) / L_  # + = outward
            # the side may sit on the bright kiosk body with the whole bezel band inside it: then the screen edge
            # can be well inside (a popup's dimmed margin never looks like that: there the side sits on dark)
            jp = int(np.argmin(np.abs(idx - pos)))
            on_body = j1 < jp and np.nanmedian(pr[jp:jp + 3]) >= 0.5 * bright
            if out < -(max_in_body if on_body else max_in) or out > max_out:
                part_notes.append(f"kept ({out * 100:+.1f}%)"); continue
            pts.append(((a0 + a1) / 2, float(idx[cut])))
            part_notes.append(f"{out * 100:+.1f}%")
        notes[sd] = " / ".join(part_notes)
        if not pts:
            continue
        if len(pts) == 1:
            pts.append((pts[0][0] + 1.0, pts[0][1]))  # one part only: keep the side's direction
        (a1_, p1_), (a2_, p2_) = pts[0], pts[-1]
        # line through the parts' edge points, as (point, direction) in frontal coordinates
        if horiz:
            lines[sd] = (np.array([a1_, p1_]), np.array([a2_ - a1_, p2_ - p1_]))
        else:
            lines[sd] = (np.array([p1_, a1_]), np.array([p2_ - p1_, a2_ - a1_]))
    default = {"T": (np.array([0, my]), np.array([1.0, 0])), "B": (np.array([0, my + h]), np.array([1.0, 0])),
               "L": (np.array([mx, 0]), np.array([0, 1.0])), "R": (np.array([mx + w, 0]), np.array([0, 1.0]))}
    Lns = {sd: lines.get(sd, default[sd]) for sd in "TRBL"}

    def cross(l1, l2):
        (a, u), (b, v) = l1, l2
        A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
        return None if abs(np.linalg.det(A)) < 1e-9 else a + np.linalg.solve(A, b - a)[0] * u

    corners = [cross(Lns["T"], Lns["L"]), cross(Lns["T"], Lns["R"]), cross(Lns["B"], Lns["R"]), cross(Lns["B"], Lns["L"])]
    if any(c is None for c in corners):
        return quad, {"all": "rejected (parallel sides)"}
    rect = np.float32(corners)
    if not cv2.isContourConvex(rect) or cv2.contourArea(rect) < 0.5 * w * h:
        return quad, {"all": "rejected (shape)"}
    back = cv2.perspectiveTransform(rect[None], np.linalg.inv(M))[0]
    return [[round(float(x / W_), 4), round(float(y / H_), 4)] for x, y in back], notes


def _det_attempt(img, jpeg, rough, W, H):
    """One detector-box attempt: SAM candidates -> per-corner consensus -> bezel snapping on close-ups,
    plus the checks used to judge it (see _detect_det)."""
    t = time.time()
    cands = sam_candidates(jpeg, rough, grid=True)
    ms = {"sam": round((time.time() - t) * 1000)}
    qs = [c for c in cands if "quad" in c and crop.iou(c["quad"], rough) >= 0.3]
    tried = [f"{c['name']}{c['pad']}: score={c['score']:.2f} agree={crop.iou(c['quad'], rough):.2f}" for c in qs]
    a = {"rough": [list(p) for p in rough], "quad": [list(p) for p in rough], "used": "det", "tried": tried, "ms": ms}
    if len(qs) >= 2:
        a["quad"], a["used"] = consensus([c["quad"] for c in qs], W, H), "consensus"
    elif qs:
        a["quad"], a["used"] = qs[0]["quad"], qs[0]["name"]
    share = crop.area(rough)
    a["gated"] = bool(qs) and BEZEL_SNAP_AREA[0] <= share <= BEZEL_SNAP_AREA[1]
    a["sides"] = 0
    if a["gated"]:
        snap_fn = bezel_snap_outer if BEZEL_MODE == "outer" else bezel_snap
        t = time.time(); q2, moved = snap_fn(img, a["quad"]); ms["snap"] = round((time.time() - t) * 1000)
        tried.append(f"bezel snap ({BEZEL_MODE}): " + ("".join("TRBL"[i] if m else "-" for i, m in enumerate(moved)) if any(moved) else "no clear bezel"))
        if any(moved):
            a["quad"], a["snapped"], a["sides"] = q2, "bezel", int(sum(moved))
    else:
        tried.append(f"bezel snap: skipped (box covers {share:.0%} of the photo)")
    # checks: on a close-up, finding the bezel border on fewer than 2 sides means the shape was not confirmed;
    # corners that cannot be right angles in 3D mean the quad is not the display (rect_error)
    a["touch"] = sum(1 for x, y in a["quad"] if min(x, y) < BORDER_EPS or max(x, y) > 1 - BORDER_EPS)
    a["rect_err"] = rect_error(a["quad"], W, H)
    t = time.time(); a["tilt"], _ = rectified_tilt(img, a["quad"]); ms["check"] = round((time.time() - t) * 1000)
    # (a corner on the photo border is NOT a reason: close-ups often frame the screen edge to edge)
    reasons = (["bezel border found on fewer than 2 sides"] if a["gated"] and a["sides"] < 2 else []) + \
              (["corners are not right angles in 3D"] if a["rect_err"] > RECT_ERR_MAX else []) + \
              (["lines in the rectified screen are tilted"] if a["tilt"] is not None and a["tilt"] > TILT_MAX_DEG else [])
    a["valid"], a["reasons"] = not reasons, reasons
    return a


BORDER_EPS = 0.005
# |cos| of the 3D corner angle above which a quad cannot be the photographed display (good results stayed under
# 0.06; failures that ran onto the body diagonally scored 0.12-0.17). Photos cropped off-centre break the
# image-centre assumption and can score up to ~0.17 even when right, hence only used as a warning + retry signal.
RECT_ERR_MAX = 0.12
# A failed check retries the detector's other top boxes. First tried 2026-11-18 it made hakik worse (runner-up
# bigger, passed while including more body: 9 -> 34 HD95) and was switched off; hakik was later dropped from the
# set, and commons_spirit (top box = the card reader's small screen, runner-up = the monitor) needs it. Results
# still failing after the retry are reported (validation.ok = false: ask the user to confirm the corners).
RETRY_RUNNER_UP = True
TILT_MAX_DEG = 2.0  # leftover tilt (deg) inside the flattened screen above which the result is flagged; straight results are ~0.3-1.2
_lsd = None


def rectified_tilt(img, quad, side=800):
    """Flatten img with quad and measure how far its long straight lines are from horizontal/vertical (degrees,
    length-weighted). Everything drawn on the screen (popup, buttons, menu grid, text rows) lies in the display's
    plane, so after a correct rectification those lines are axis-aligned; a skewed result leaves them tilted.
    -> (degrees or None when there are too few lines, number of lines used)"""
    global _lsd
    if _lsd is None:
        _lsd = cv2.createLineSegmentDetector()
    flat, _ = warp(img, quad, aspect=True, long_side=side)
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    m = int(0.04 * min(h, w))
    segs = _lsd.detect(g[m:h - m, m:w - m])[0]  # skip the frame edge itself
    if segs is None:
        return None, 0
    segs = segs[:, 0]
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = L > 0.08 * min(h, w)
    segs, L = segs[keep], L[keep]
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    dev = np.minimum(ang, np.minimum(np.abs(ang - 90), 180 - ang))
    near = dev < 25  # lines meant to be horizontal or vertical
    if near.sum() < 4:
        return None, int(near.sum())
    return float(np.average(dev[near], weights=L[near])), int(near.sum())


def rect_error(q, W, H, f_range=(0.45, 3.0), f_assumed=0.75):
    """How far quad q is from a photographed rectangle, as |cos| of the 3D corner angle (0 = right angles).
    Image angles are not 90 deg under perspective, so the quad is lifted back to 3D: the two vanishing points
    give the focal length (Zhang & He); if that is not phone-like (f_range x long side) a typical f is assumed
    and the angle between the back-projected edge directions is measured. A display seen from any angle scores ~0;
    a quad that ran onto the body or cut diagonally through the screen usually cannot be a rectangle."""
    try:
        q_px = [(x * W, y * H) for x, y in q]
        tl, tr, br, bl = [np.array([p[0], p[1], 1.0]) for p in q_px]
        m1, m2, m3, m4 = tl, tr, bl, br
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2, n3 = k2 * m2 - m1, k3 * m3 - m1
        u0, v0, L = W / 2, H / 2, max(W, H)
        f = None
        if abs(n2[2] * n3[2]) > 1e-9:
            f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0) +
                   (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
            if np.isfinite(f2) and f2 > 0 and f_range[0] * L < np.sqrt(f2) < f_range[1] * L:
                f = np.sqrt(f2)
        if f is None:
            f = f_assumed * L
        Ai = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]))
        B = Ai.T @ Ai
        c = abs(n2 @ B @ n3) / np.sqrt(abs(n2 @ B @ n2) * abs(n3 @ B @ n3))
        return float(c) if np.isfinite(c) else 1.0
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError, ValueError):
        return 1.0


def _severity(a):
    """How far an attempt misses the checks, in units of each limit (<= 1 = within limits). rect_err alone cannot
    rank failures: any quad whose implied focal length is phone-like scores 0 (e.g. a card reader's small screen),
    so the leftover tilt counts too."""
    return max(a["rect_err"] / RECT_ERR_MAX, (a["tilt"] or 0.0) / TILT_MAX_DEG,
               1.01 if "bezel border found on fewer than 2 sides" in a["reasons"] else 0.0)


def _detect_det(img, jpeg, W, H):
    """det_sam on the detector's best box, with checks reported in validation (ok / reasons / rect_err).
    With RETRY_RUNNER_UP a failed check is retried on the detector's other top boxes (~0.4s each, stops at the
    first that passes); when every attempt fails, the one that misses the limits by the least is kept."""
    t = time.time()
    boxes = crop.det_quads(jpeg)
    ms = {"det": round((time.time() - t) * 1000)}
    if not boxes:
        return {"quad": None, "used": None, "tried": ["det: none"], "ms": ms}
    first = _det_attempt(img, jpeg, boxes[0], W, H)
    if SECOND_PASS and first["tilt"] is not None and (first["tilt"] > SECOND_PASS_TILT or first["rect_err"] > RECT_ERR_MAX):
        t = time.time()
        cur, cur_tilt, notes = first["quad"], first["tilt"], []
        for _ in range(2):  # a second round when one straightening leaves the lines still tilted
            q2, note = second_pass(img, cur)
            if q2 is None:
                notes.append(note); break
            options = [(q2, note)]
            if first["gated"]:
                snap_fn = bezel_snap_outer if BEZEL_MODE == "outer" else bezel_snap
                q3, moved = snap_fn(img, q2)
                if any(moved):
                    options.append((q3, note + "; bezel snap " + "".join("TRBL"[i] if m else "-" for i, m in enumerate(moved))))
            scored = [(rectified_tilt(img, q)[0], q, nt) for q, nt in options]
            scored = [x for x in scored if x[0] is not None and rect_error(x[1], W, H) <= max(first["rect_err"], RECT_ERR_MAX)]
            if not scored:
                notes.append(note + " (no usable result)"); break
            tl2, q2, nt = min(scored, key=lambda x: x[0])
            if tl2 >= cur_tilt - 0.5:
                notes.append(f"{nt}; tilt {cur_tilt:.1f} -> {tl2:.1f} deg (not better, dropped)"); break
            notes.append(f"{nt}; tilt {cur_tilt:.1f} -> {tl2:.1f} deg (kept)")
            cur, cur_tilt = q2, tl2
            if cur_tilt <= TILT_MAX_DEG:
                break
        first["tried"].insert(0, "second pass: " + " | ".join(notes))
        if cur is not first["quad"]:
            rect2 = rect_error(cur, W, H)
            reasons = [r for r in first["reasons"] if r == "bezel border found on fewer than 2 sides"] + \
                      (["corners are not right angles in 3D"] if rect2 > RECT_ERR_MAX else []) + \
                      (["lines in the rectified screen are tilted"] if cur_tilt > TILT_MAX_DEG else [])
            first.update(quad=cur, used=first["used"] + "+second_pass", tilt=cur_tilt, rect_err=rect2,
                         valid=not reasons, reasons=reasons)
        first["ms"]["second_pass"] = round((time.time() - t) * 1000)
    if FRONTAL_EXTENT and FRONTAL_AREA[0] <= crop.area(boxes[0]) <= FRONTAL_AREA[1]:
        # last step on close-ups: re-decide where each side ends in a frontal view (rows/columns at once)
        t = time.time(); q4, notes = frontal_extent(img, first["quad"]); first["ms"]["extent"] = round((time.time() - t) * 1000)
        first["tried"].insert(0, "frontal extent: " + ", ".join(f"{k} {v}" for k, v in notes.items()))
        if q4 != first["quad"]:
            first["quad"] = q4
    attempts, tried_boxes = [first], [boxes[0]]
    if RETRY_RUNNER_UP and not first["valid"]:
        # e.g. the detector's top box is a card reader's small display and the main screen is the runner-up
        for b in boxes[1:]:
            if any(crop.iou(b, o) >= 0.9 for o in tried_boxes):
                continue
            tried_boxes.append(b)
            attempts.append(_det_attempt(img, jpeg, b, W, H))
            if attempts[-1]["valid"]:
                break
    best = max(attempts, key=lambda a: (a["valid"], -_severity(a)))  # ties keep the first attempt
    for k, a in enumerate(attempts):  # retry_* = summed over all retried boxes
        tag = "" if k == 0 else "retry_"
        for key, v in a["ms"].items():
            ms[tag + key] = ms.get(tag + key, 0) + v if k else v
    tried = list(best["tried"])
    if len(attempts) > 1:
        k = attempts.index(best)
        tried.insert(0, f"validation: first box failed ({'; '.join(first['reasons'])}) -> retried with {len(attempts) - 1} more"
                        f" detector box(es); severity " + " / ".join(f"{_severity(a):.2f}" for a in attempts)
                        + (f", kept box {k + 1}" if k else ", kept the first"))
    out = {k: best[k] for k in ("rough", "quad", "used") if k in best}
    if best.get("snapped"):
        out["snapped"] = best["snapped"]
    out["validation"] = {"ok": best["valid"], "reasons": best["reasons"], "retried": len(attempts) > 1,
                         "rect_err": round(best["rect_err"], 3),
                         "tilt_deg": None if best["tilt"] is None else round(best["tilt"], 2)}
    return {**out, "tried": tried, "ms": ms}


def detect(img, jpeg, method="det_sam"):
    """Find the display in img (jpeg = the same photo's bytes for the detector/SAM server). method is kept for
    old callers; only "det_sam" exists."""
    H, W = img.shape[:2]
    return _detect_det(img, jpeg, W, H)


def aspect_ratio(q_px, W, H, f_assumed=None):
    """True width/height of the rectangle seen as quad (tl,tr,br,bl px), principal point = image center.
    Focal length comes from the quad itself when it has two vanishing points; when it doesn't (e.g. a pure
    sideways turn keeps top/bottom parallel) a typical phone main camera is assumed (f ~ 0.75 x long side).
    Returns (ratio, source) or (None, reason)."""
    tl, tr, br, bl = [np.array([p[0], p[1], 1.0]) for p in q_px]
    m1, m2, m3, m4 = tl, tr, bl, br  # paper order: (0,0) (w,0) (0,h) (w,h)
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    u0, v0 = W / 2, H / 2
    f, src = None, "perspective"
    if abs(n2[2] * n3[2]) > 1e-9:
        f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0) +
               (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
        if np.isfinite(f2) and f2 > 0 and 0.3 * max(W, H) < np.sqrt(f2) < 4 * max(W, H):
            f = np.sqrt(f2)
    if f is None:
        f, src = (f_assumed or 0.75 * max(W, H)), "assumed_f"
    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
    Ai = np.linalg.inv(A)
    B = Ai.T @ Ai
    r2 = (n2 @ B @ n2) / (n3 @ B @ n3)
    if not np.isfinite(r2) or r2 <= 0:
        return None, "degenerate"
    r = float(np.sqrt(r2))
    return (r, src) if 0.2 < r < 5 else (None, "out_of_range")


def warp(img, quad, aspect=True, long_side=1600):
    H, W = img.shape[:2]
    q = np.float32([[x * W, y * H] for x, y in quad])
    tl, tr, br, bl = q
    w_e = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h_e = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    r, src = aspect_ratio(q, W, H) if aspect else (None, "off")
    ratio = float(r if r else w_e / max(h_e, 1))  # norm() of float32 points is np.float32: not JSON-serialisable
    if ratio >= 1:
        w, h = long_side, int(round(long_side / ratio))
    else:
        w, h = int(round(long_side * ratio)), long_side
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"ratio": round(ratio, 3), "ratio_from": src if r else "edges",
                 "edge_ratio": round(float(w_e / max(h_e, 1)), 3)}


def b64jpeg(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()
