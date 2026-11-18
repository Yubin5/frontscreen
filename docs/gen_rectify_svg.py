"""Builds docs/rectify_pipeline.svg (the README figure) from the output of docs/fig_stages.py.

  python docs/gen_rectify_svg.py data/real/test_05.jpg data/real/commons_spirit.jpg <fig_dir> docs/fig_numbers.json > docs/rectify_pipeline.svg
  fig_dir     : stages.json, flat_check.jpg, frontal.jpg, warped.jpg (from fig_stages.py)
  numbers.json: {"groups": [[label, n, iou, hd95, cut, over]], "cut_over_2": int, "sec": float, "test_n": int}
PNG: open the SVG in a browser (headless Chrome screenshot at 2x).
"""
import base64, json, os, sys
import cv2, numpy as np

ex_path, retry_path, fig, num_path = sys.argv[1:5]
st = json.load(open(os.path.join(fig, "stages.json")))
nums = json.load(open(num_path))
img = cv2.imread(ex_path)
H, W = img.shape[:2]
IW, IH = 260, 195


def b64(im, w):
    im = cv2.resize(im, (w, max(1, int(round(im.shape[0] * w / im.shape[1])))), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


# stages 1-4: one crop around the screen, same aspect as the panel, shared by the four panels
xs, ys = [p[0] * W for p in st["rough"]], [p[1] * H for p in st["rough"]]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
cw = max((max(xs) - min(xs)) * 1.18, (max(ys) - min(ys)) * 1.18 * IW / IH)
ch = cw * IH / IW
x0c, y0c = max(0, cx - cw / 2), max(0, cy - ch / 2)
x0c, y0c = min(x0c, max(0, W - cw)), min(y0c, max(0, H - ch))
crop = img[int(y0c):int(y0c + ch), int(x0c):int(x0c + cw)]
CROP = b64(crop, 2 * IW)


def to_panel(q, px, py):  # fractions of the photo -> panel coordinates
    return " ".join(f"{px + (x * W - x0c) / cw * IW:.1f},{py + (y * H - y0c) / ch * IH:.1f}" for x, y in q)


def fit(im):  # letterbox an image into the panel -> (data uri, x, y, w, h) relative to the panel
    k = min(IW / im.shape[1], IH / im.shape[0])
    w, h = im.shape[1] * k, im.shape[0] * k
    return b64(im, int(2 * w)), (IW - w) / 2, (IH - h) / 2, w, h


FONT = "'Pretendard','Noto Sans KR','Noto Sans CJK KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif"
TOT_H = 1265
o = []
add = o.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 {TOT_H}" width="1600" height="{TOT_H}" font-family="{FONT}">')
add("""<style>
.t1{font-size:30px;font-weight:700;fill:#1b1d21}.sub{font-size:15px;fill:#5f6670}
.h{font-size:17px;font-weight:700;fill:#1b1d21}.b{font-size:13px;fill:#3d4552}.m{font-size:12.5px;fill:#6b737e}
.s{font-size:12px;fill:#6b737e}.v{font-size:12px;font-weight:700;fill:#1b1d21}.chip{font-size:11px;font-weight:700;letter-spacing:.5px}
.big{font-size:28px;font-weight:700;fill:#1b1d21}.card{fill:#fff;stroke:#cfd4db;stroke-width:1.2}
.fix{fill:#fffaf3;stroke:#e8c9a0;stroke-width:1.2}.panel{fill:#fff;stroke:#dde1e7}
.arrow{stroke:#8a929c;stroke-width:2;fill:none;marker-end:url(#ah)}.darrow{stroke:#c47a2c;stroke-width:2;fill:none;stroke-dasharray:6 4;marker-end:url(#ah2)}
.lab{font-size:11px;font-weight:700}
</style>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a929c"/></marker>
<marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#c47a2c"/></marker></defs>""")
add(f'<rect width="1600" height="{TOT_H}" fill="#fff"/>')
add('<text x="40" y="58" class="t1">디스플레이 화면 보정 파이프라인</text>')
add('<text x="40" y="88" class="sub">사진 속 키오스크 화면을 찾아 정면으로 펴고, 글씨까지 읽기 · 학습 없이 공개 모델 조합 + 기하 계산 · '
    f'로컬 GPU 한 장, 펴기 약 {nums["sec"]:.1f}초 (+ OCR 약 0.5~1.3초)</text>')

CX, CW, CH = [40, 347, 654, 961, 1268], 292, 400
ROW = [112, 572]
r0 = st["retry"]
cards = [
    # (row, col, title, chip, body lines, note lines, style)
    (0, 0, "① 검출기 상자", "GPU", ["Grounding DINO tiny에", "'touch screen' 상자를 요청"], ["384px 입력 · 약 0.3초", "상위 3개 상자를 보관 (⑥에서 사용)"], "card"),
    (0, 1, "② SAM 후보 5개", "GPU", ["상자 + 안쪽(+)·바깥(−) 점으로", "여백을 바꿔 가며 5번 분할"], ["사진 인코딩은 한 번 · 약 0.4초", "후보마다 경계가 조금씩 다름"], "card"),
    (0, 2, "③ 꼭짓점 합의", "CPU", ["꼭짓점마다 후보들의", "중간 위치를 사용"], ["잘린 후보와 넘친 후보가", "서로 상쇄돼 결과가 안정적"], "card"),
    (0, 3, "④ 베젤 경계 맞추기", "CPU", ["베젤 바깥 테두리 안쪽의", "가장 바깥 경계로 변을 이동"], ["근접 사진(상자 35~80%)만", "변마다 48곳 · RANSAC 직선"], "card"),
    (0, 4, "⑤ 결과 검사", "CPU", ["3D 직각: 소실점→초점거리로", "되돌린 모서리 |cos| ≤ 0.12",
                                 "펼친 화면 속 직선 기울기 ≤ 2°"],
     [f"이 사진: 직각 오차 {st['rect_err']:.2f} · 기울기 {st['tilt']:.1f}° → 통과",
      "초록 = 수평·수직 직선, 빨강 = 기울어진 직선"], "card"),
    (1, 0, "⑥ 걸리면: 보정·재시도", "CPU+GPU", ["2차 보정: 화면 속 직선의 소실점으로", "사각형을 바로잡기 (최대 2회)",
                                              "그래도 걸리면 검출기의 다른 상자로"],
     [f"기준 초과가 가장 작은 결과를 고름", f"예: 카드 리더 {r0[0]['severity']:.2f} → 모니터 {r0[1]['severity']:.2f}"], "fix"),
    (1, 1, "⑦ 정면에서 변 끝 다시 정하기", "CPU", ["8% 여백을 두고 정면으로 편 뒤", "변마다 밝기 단면으로 화면 끝 결정"],
     ["한 번에 안쪽 5% · 바깥 2%까지만", "팝업이 떠도 화면 전체를 유지"], "card"),
    (1, 2, "⑧ 원근 보정", "CPU", ["초점거리 추정 → 실제 가로세로비", "→ 정면 이미지 (긴 변 1600px)"],
     [f"이 사진: {st['aspect']['ratio']:.2f} (보이는 변 {st['aspect']['edge_ratio']:.2f})", "못 구하면 f = 0.75 × 사진 긴 변"], "card"),
    (1, 3, "⑨ 글씨 읽기 (OCR)", "CPU", ["PaddleOCR 모바일 검출 + 한국어", "인식, 위→아래 줄 순서로 정리"],
     [f"이 사진: {len(st['ocr'])}개 · 한·영·숫자", "일본어·중국어는 못 읽음"], "card"),
]
for row, col, title, chip, lines, notes, style in cards:
    x, y = CX[col], ROW[row]
    add(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="12" class="{style}"/>')
    add(f'<text x="{x + 16}" y="{y + 32}" class="h">{title}</text>')
    gpu = "GPU" in chip
    cwid = 14 + 7.2 * len(chip)
    add(f'<rect x="{x + CW - 14 - cwid}" y="{y + 16}" width="{cwid}" height="22" rx="11" fill="{"#fbe3d3" if gpu else "#e9edf2"}"/>'
        f'<text x="{x + CW - 14 - cwid / 2}" y="{y + 31}" class="chip" text-anchor="middle" fill="{"#a24f1c" if gpu else "#4a5260"}">{chip}</text>')
    px, py = x + 16, y + 54
    add(f'<rect x="{px - 1}" y="{py - 1}" width="{IW + 2}" height="{IH + 2}" rx="5" fill="#f3f5f8"/>')
    k = (row, col)
    if row == 0 and col < 4:
        add(f'<image href="{CROP}" x="{px}" y="{py}" width="{IW}" height="{IH}" preserveAspectRatio="none"/>')
        if col == 0:
            add(f'<polygon points="{to_panel(st["rough"], px, py)}" fill="none" stroke="#f2b01e" stroke-width="3"/>')
        if col == 1:
            for q in st["cands"]:
                add(f'<polygon points="{to_panel(q, px, py)}" fill="none" stroke="#fff" stroke-width="1.4" opacity="0.95"/>')
            for p in st["pos"]:
                ax, ay = map(float, to_panel([p], px, py).split(","))
                add(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="#1f6fd1" stroke="#fff" stroke-width="2"/>'
                    f'<path d="M{ax - 3.5:.1f},{ay:.1f} H{ax + 3.5:.1f} M{ax:.1f},{ay - 3.5:.1f} V{ay + 3.5:.1f}" stroke="#fff" stroke-width="1.8"/>')
            for p in st["neg"]:
                ax, ay = map(float, to_panel([p], px, py).split(","))
                if px <= ax <= px + IW and py <= ay <= py + IH:
                    add(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="#fff" stroke="#1b1d21" stroke-width="2"/>'
                        f'<path d="M{ax - 3.5:.1f},{ay:.1f} H{ax + 3.5:.1f}" stroke="#1b1d21" stroke-width="1.8"/>')
        if col == 2:
            for q in st["cands"]:
                add(f'<polygon points="{to_panel(q, px, py)}" fill="none" stroke="#fff" stroke-width="1" opacity="0.7"/>')
            add(f'<polygon points="{to_panel(st["consensus"], px, py)}" fill="none" stroke="#d63fd6" stroke-width="3"/>')
        if col == 3:
            add(f'<polygon points="{to_panel(st["consensus"], px, py)}" fill="none" stroke="#d63fd6" stroke-width="1.5" stroke-dasharray="5 4"/>')
            add(f'<polygon points="{to_panel(st["snap"], px, py)}" fill="rgba(226,59,47,0.10)" stroke="#e23b2f" stroke-width="3"/>')
    else:
        if k == (0, 4):
            im = cv2.imread(os.path.join(fig, "flat_check.jpg"))
        elif k == (1, 0):
            im = cv2.imread(retry_path)
            h2, w2 = im.shape[:2]
            for a, col_ in zip(r0[:2], [(47, 59, 226), (60, 190, 60)]):
                cv2.polylines(im, [np.int32([[u * w2, v * h2] for u, v in a["quad"]])], True, col_, max(4, w2 // 250))
        elif k == (1, 1):
            im = cv2.imread(os.path.join(fig, "frontal.jpg"))
            cv2.polylines(im, [np.int32(st["frontal_before"])], True, (226, 59, 226), 2, cv2.LINE_AA)
            cv2.polylines(im, [np.int32(st["frontal_after"])], True, (47, 59, 226), 3, cv2.LINE_AA)
        elif k == (1, 2):
            im = cv2.imread(os.path.join(fig, "warped.jpg"))
        else:
            im = cv2.imread(os.path.join(fig, "warped.jpg"))
            for l in st["ocr"]:
                x1, y1, x2, y2 = l["box"]
                cv2.rectangle(im, (x1, y1), (x2, y2), (42, 101, 196), 3)
        uri, ox, oy, w, h = fit(im)
        add(f'<image href="{uri}" x="{px + ox:.1f}" y="{py + oy:.1f}" width="{w:.1f}" height="{h:.1f}"/>')
        if k == (1, 0):  # labels in the letterbox bands above / below the wide photo
            add(f'<text x="{px + 6}" y="{py + oy - 7:.1f}" class="lab" fill="#e23b2f">✕ 1순위 상자: 카드 리더 (검사 실패)</text>'
                f'<text x="{px + 6}" y="{py + oy + h + 15:.1f}" class="lab" fill="#2f9e44">✓ 재시도한 상자: 모니터</text>')
    ty = y + 54 + IH + 30
    for j, l in enumerate(lines):
        add(f'<text x="{x + 16}" y="{ty + j * 21}" class="b">{l}</text>')
    for j, l in enumerate(notes):
        add(f'<text x="{x + 16}" y="{ty + len(lines) * 21 + 12 + j * 19}" class="m">{l}</text>')
    if col < 4 and not (row == 1 and col == 3):
        add(f'<path d="M{x + CW + 1},{y + 54 + IH / 2} H{CX[col + 1] - 1}" class="arrow"/>')
add(f'<path d="M{CX[3] + CW + 1},{ROW[1] + 54 + IH / 2} H{CX[4] - 1}" class="arrow"/>')

# output card (row 2, col 4)
x, y = CX[4], ROW[1]
add(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="12" fill="#f3f5f8" stroke="#cfd4db"/>')
add(f'<text x="{x + 16}" y="{y + 32}" class="h">출력 (HTTP API)</text>')
for j, (a, b) in enumerate([("POST /flatten", "사진 → 펴진 화면 JPEG"), ("POST /read", "사진 → 글자 + 위치 JSON"),
                            ("확인 필요 표시", "검사에 끝까지 걸리면 doubtful")]):
    yy = y + 86 + j * 74
    add(f'<text x="{x + 20}" y="{yy}" class="h" style="font-size:15px">{a}</text><text x="{x + 20}" y="{yy + 24}" class="b">{b}</text>')
add(f'<text x="{x + 20}" y="{y + 330}" class="m">꼭짓점을 손으로 고치면 ⑧부터 다시</text>')
add(f'<text x="{x + 20}" y="{y + 350}" class="m">(실험용 테스트 페이지)</text>')

# connector: ⑤ -> ⑥ (only when a check fails), passing straight on to ⑦ otherwise
y5 = ROW[0] + CH
ym = (y5 + ROW[1]) / 2
add(f'<path d="M{CX[4] + CW / 2},{y5 + 1} V{ym} H{CX[0] + CW / 2} V{ROW[1] - 1}" class="darrow"/>')
add(f'<text x="{CX[0] + CW / 2 + 10}" y="{ym - 8}" class="s" fill="#a2602a">검사에 걸릴 때만 ⑥ · 통과하면 바로 ⑦로</text>')

# legend
LY = ROW[1] + CH + 30
lx = 40
for col, kind, label in [("#f2b01e", "line", "검출기 상자"), ("#1f6fd1", "dot", "안쪽 점 (+)"), ("#1b1d21", "ring", "바깥 점 (−)"),
                         ("#c9ced6", "line", "SAM 후보"), ("#d63fd6", "line", "후보 합의"), ("#e23b2f", "line", "최종 화면 영역"),
                         ("#c4652a", "line", "OCR 글자 상자")]:
    if kind == "line":
        add(f'<line x1="{lx}" y1="{LY - 4}" x2="{lx + 20}" y2="{LY - 4}" stroke="{col}" stroke-width="3"/>')
    elif kind == "dot":
        add(f'<circle cx="{lx + 8}" cy="{LY - 4}" r="6" fill="{col}"/>')
    else:
        add(f'<circle cx="{lx + 8}" cy="{LY - 4}" r="6" fill="#fff" stroke="{col}" stroke-width="2"/>')
    add(f'<text x="{lx + 26}" y="{LY}" class="s">{label}</text>')
    lx += 40 + 12 * len(label) + 30

# bottom: results + open problems
PY = LY + 26
add(f'<rect x="40" y="{PY}" width="760" height="190" rx="16" class="panel"/>')
add(f'<text x="60" y="{PY + 32}" class="h">개발셋 결과 (정답은 손으로 표시 · 화면 대각선 대비 %)</text>')
tiles = [(f"{g[3]:.1f}%", f"HD95 · {g[0]} {g[1]}장", f"IoU {g[2]:.2f} · 잘림 {g[4]:.1f} · 넘침 {g[5]:.1f}") for g in nums["groups"]]
tiles.append((f"{nums['cut_over_2']}장", "화면이 2% 넘게 잘린 사진", f"처리 시간 중앙값 {nums['sec']:.1f}초"))
for j, (big, a, b) in enumerate(tiles):
    tx = 60 + j * 245
    add(f'<rect x="{tx}" y="{PY + 48}" width="230" height="110" rx="10" fill="#f3f5f8"/>')
    add(f'<text x="{tx + 16}" y="{PY + 88}" class="big">{big}</text><text x="{tx + 16}" y="{PY + 116}" class="b">{a}</text>'
        f'<text x="{tx + 16}" y="{PY + 138}" class="m">{b}</text>')
add(f'<text x="60" y="{PY + 178}" class="m">상수는 이 사진들을 보며 정했다 · 처음 보는 기계용 테스트셋 {nums["test_n"]}장(Commons)은 정답 표시 중</text>')
RX = 830
add(f'<rect x="{RX}" y="{PY}" width="730" height="190" rx="16" class="panel"/>')
add(f'<text x="{RX + 20}" y="{PY + 32}" class="h">남은 문제</text>')
for j, (a, b) in enumerate([("강한 창문 반사 + 팝업", "화면 아래쪽이 하얗게 날아가면 경계와 글씨가 약해짐"),
                            ("화면 일부가 사진 밖으로 나간 측면 사진", "아래 변이 대각선으로 잡힘 (버튼 줄이 잘림)"),
                            ("광각 렌즈 · 크게 내려다본 사진", "결과가 맞아도 직각 검사에 걸려 '확인 필요'가 뜰 수 있음")]):
    yy = PY + 66 + j * 42
    add(f'<circle cx="{RX + 26}" cy="{yy - 5}" r="3" fill="#6b737e"/><text x="{RX + 36}" y="{yy}" class="b">{a}</text>'
        f'<text x="{RX + 36}" y="{yy + 18}" class="m">{b}</text>')
add(f'<text x="40" y="{TOT_H - 16}" class="s">예시 사진: Wikimedia Commons — 小田急4000系ファン 「Hankyu Mukonoso station Ticket vending machine screen」 (CC0), '
    '42-BRT 「Disused Spirit Airlines check-in counter at DTW Evans Terminal…」 (CC BY-SA 4.0) · 2026-11-18 기준</text>')
add("</svg>")
print("\n".join(o))
