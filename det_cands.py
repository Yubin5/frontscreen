"""Cache detector+SAM candidates (rough box from Grounding DINO, every SAM candidate) for offline selection work.
  python det_cands.py   -> results/cands_det.json"""
import json, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rectify, screen_bench as sb

LAB = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(LAB, "results", "cands_det.json")
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
for path, _ in sb.cases():
    if path in cache:
        continue
    jpeg = open(os.path.join(LAB, "data", path), "rb").read()
    img = cv2.imread(os.path.join(LAB, "data", path))
    rough, cands, ms = rectify.candidates(img, jpeg, "det_sam")
    cache[path] = {"rough": rough and [list(p) for p in rough], "cands": cands, "ms": ms}
    json.dump(cache, open(OUT, "w"))
print(len(cache), "cached")
