"""Text on the flattened screen: PaddleOCR (PP-OCRv5 mobile detector + Korean recognizer, CPU).

read(img) -> {"text": rows joined top to bottom, "lines": [{"text", "score", "box": [x1,y1,x2,y2], "row"}], "ms"}
Boxes are pixels in img (the flattened screen). The Korean recognizer also reads digits and Latin letters,
but not Japanese/Chinese.

Notes from the 2026-11-09 tests: the mobile detector is as accurate as the server one and 5x faster;
naming a detector model makes PaddleOCR drop lang=, so the Korean recognizer is named too; Paddle predictors
break when called from several threads, so every call runs on one worker thread.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_pool = ThreadPoolExecutor(max_workers=1)
_model = None


def _get():
    global _model
    if _model is None:
        from paddleocr import PaddleOCR  # heavy import: only when OCR is used
        _model = PaddleOCR(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False, text_detection_model_name="PP-OCRv5_mobile_det",
                           text_recognition_model_name="korean_PP-OCRv5_mobile_rec")
    return _model


def warm():
    """Load the models now (a few seconds) instead of on the first request."""
    _pool.submit(_get).result()


def group_rows(items):
    """Group boxes into text rows (vertical overlap with the row's running band), each row left to right."""
    rows = []
    for it in sorted(items, key=lambda it: (it["box"][1] + it["box"][3]) / 2):
        x1, y1, x2, y2 = it["box"]
        for r in rows:
            top, bot = r["band"]
            if min(y2, bot) - max(y1, top) > 0.5 * min(y2 - y1, bot - top):
                r["items"].append(it); r["band"] = (min(top, y1), max(bot, y2)); break
        else:
            rows.append({"band": (y1, y2), "items": [it]})
    return [sorted(r["items"], key=lambda it: it["box"][0]) for r in rows]


def read(img, min_score=0.5):
    t = time.time()
    r = _pool.submit(lambda: _get().predict(img)[0]).result()
    items = [{"text": tx, "score": round(float(s), 3), "box": [int(v) for v in b]}
             for tx, s, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"])
             if tx.strip() and s >= min_score]
    rows = group_rows(items)
    return {"text": "\n".join(" ".join(it["text"] for it in row) for row in rows),
            "lines": [{**it, "row": i} for i, row in enumerate(rows) for it in row],
            "ms": round((time.time() - t) * 1000)}
