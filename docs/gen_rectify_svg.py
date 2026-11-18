"""Builds docs/rectify_pipeline.svg for the detector+SAM pipeline (det_sam).

Run inside the kiosk-lab image:  python docs/gen_rectify_svg.py <example.jpg> <stages.json> <warped.jpg> <numbers.json>
  stages.json : rough, cands, consensus, snap, pos, neg (fractions) from rectify on the example photo
  numbers.json: {"groups": [[label, n], ...], "methods": {name: {"hd": [...], "cuts": int, "sec": float}}}
"""
import base64, json, sys
import cv2, numpy as np

ex_path, st_path, warp_path, num_path = sys.argv[1:5]
img = cv2.imread(ex_path)
st = json.load(open(st_path))
warped = cv2.imread(warp_path)
nums = json.load(open(num_path))


def b64(im, w):
    im = cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


A, WP = b64(img, 560), b64(warped, 560)
FONT = "'Pretendard','Noto Sans KR','Noto Sans CJK KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif"
o = []
add = o.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 990" width="1600" height="990" font-family="{FONT}">')
add("""<style>
.t1{font-size:30px;font-weight:700;fill:#1b1d21}.sub{font-size:15px;fill:#5f6670}
.h{font-size:17px;font-weight:700;fill:#1b1d21}.st{font-size:14px;font-weight:700;fill:#1b1d21}
.b{font-size:13px;fill:#3d4552}.m{font-size:12.5px;fill:#6b737e}.s{font-size:12px;fill:#6b737e}
.v{font-size:12px;font-weight:700;fill:#1b1d21}.chip{font-size:11px;font-weight:700;letter-spacing:.5px}
.big{font-size:30px;font-weight:700;fill:#1b1d21}.card{fill:#fff;stroke:#cfd4db;stroke-width:1.2}
.panel{fill:#fff;stroke:#dde1e7}.arrow{stroke:#8a929c;stroke-width:2;fill:none;marker-end:url(#ah)}
</style>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a929c"/></marker></defs>
<rect width="1600" height="990" fill="#fff"/>
<text x="40" y="58" class="t1">디스플레이 화면 보정 파이프라인</text>
<text x="40" y="88" class="sub">사진 속 키오스크 화면을 찾아 정면으로 펴기 · OCR 없이 이미지만 사용 · 학습 없이 공개 모델 조합 + 기하 계산 · 로컬 GPU 한 장, 사진 한 장당 약 0.7초</text>
""")

CX, CW, CY, CH = [40, 347, 654, 961, 1268], 292, 112, 440
IX, IY, IW, IH = 16, 54, 260, 195
cards = [
    ("① 검출기 상자", "GPU", ["Grounding DINO tiny에", "'touch screen' 상자를 요청"], ["384px 입력 · 약 0.3초", "넉넉하게 잡아 화면을 안 자름"]),
    ("② SAM 후보 5개", "GPU", ["상자 + 안쪽(+)·바깥(−) 점으로", "여백을 바꿔 가며 5번 분할"], ["사진 인코딩은 한 번 · 약 0.4초", "후보마다 경계가 조금씩 다름"]),
    ("③ 꼭짓점 합의", "CPU", ["꼭짓점마다 후보들의", "중간 위치를 사용"], ["잘린 후보와 넘친 후보가", "서로 상쇄돼 결과가 안정적"]),
    ("④ 베젤 경계 맞추기", "CPU", ["베젤 바깥 테두리 안쪽의", "가장 바깥 경계로 변을 이동"], ["근접 사진(상자 35~80%)만", "팝업이 떠도 화면 전체를 유지"]),
    ("⑤ 원근 보정", "CPU", ["초점거리 추정 → 실제 비율", "→ 정면 이미지 (긴 변 1600px)"], ["초점거리를 못 구하면", "f = 0.75 × 사진 긴 변으로 가정"]),
]
poly = lambda q, x0, y0: " ".join(f"{x0 + x * IW:.1f},{y0 + y * IH:.1f}" for x, y in q)
for i, (title, chip, lines, notes) in enumerate(cards):
    x = CX[i]
    add(f'<rect x="{x}" y="{CY}" width="{CW}" height="{CH}" rx="12" class="card"/>')
    add(f'<text x="{x + 16}" y="{CY + 32}" class="h">{title}</text>')
    gpu = chip == "GPU"
    add(f'<rect x="{x + CW - 60}" y="{CY + 16}" width="46" height="22" rx="11" fill="{"#fbe3d3" if gpu else "#e9edf2"}"/>'
        f'<text x="{x + CW - 37}" y="{CY + 31}" class="chip" text-anchor="middle" fill="{"#a24f1c" if gpu else "#4a5260"}">{chip}</text>')
    x0, y0 = x + IX, CY + IY
    add(f'<rect x="{x0 - 1}" y="{y0 - 1}" width="{IW + 2}" height="{IH + 2}" rx="5" fill="#f3f5f8"/>')
    if i < 4:
        add(f'<image href="{A}" x="{x0}" y="{y0}" width="{IW}" height="{IH}" preserveAspectRatio="none"/>')
    if i == 0:
        add(f'<polygon points="{poly(st["rough"], x0, y0)}" fill="none" stroke="#f2b01e" stroke-width="3"/>')
    if i == 1:
        for q in st["cands"]:
            add(f'<polygon points="{poly(q, x0, y0)}" fill="none" stroke="#ffffff" stroke-width="1.4" opacity="0.9"/>')
        for px, py in st["pos"]:
            cx, cy = x0 + px * IW, y0 + py * IH
            add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#1f6fd1" stroke="#fff" stroke-width="2"/>'
                f'<path d="M{cx - 3.5:.1f},{cy:.1f} H{cx + 3.5:.1f} M{cx:.1f},{cy - 3.5:.1f} V{cy + 3.5:.1f}" stroke="#fff" stroke-width="2"/>')
        for px, py in st["neg"]:
            cx, cy = x0 + px * IW, y0 + py * IH
            add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#fff" stroke="#1b1d21" stroke-width="2"/>'
                f'<path d="M{cx - 3.5:.1f},{cy:.1f} H{cx + 3.5:.1f}" stroke="#1b1d21" stroke-width="2"/>')
    if i == 2:
        for q in st["cands"]:
            add(f'<polygon points="{poly(q, x0, y0)}" fill="none" stroke="#ffffff" stroke-width="1" opacity="0.7"/>')
        add(f'<polygon points="{poly(st["consensus"], x0, y0)}" fill="none" stroke="#d63fd6" stroke-width="3"/>')
    if i == 3:
        add(f'<polygon points="{poly(st["consensus"], x0, y0)}" fill="none" stroke="#d63fd6" stroke-width="1.5" stroke-dasharray="5 4"/>')
        add(f'<polygon points="{poly(st["snap"], x0, y0)}" fill="rgba(226,59,47,0.10)" stroke="#e23b2f" stroke-width="3"/>')
    if i == 4:
        r = warped.shape[1] / warped.shape[0]
        ww = min(IW, IH * r); hh = ww / r
        add(f'<image href="{WP}" x="{x0 + (IW - ww) / 2:.1f}" y="{y0 + (IH - hh) / 2:.1f}" width="{ww:.1f}" height="{hh:.1f}"/>')
    ty = y0 + IH + 32
    for j, l in enumerate(lines):
        add(f'<text x="{x + 16}" y="{ty + j * 21}" class="b">{l}</text>')
    for j, l in enumerate(notes):
        add(f'<text x="{x + 16}" y="{ty + len(lines) * 21 + 14 + j * 19}" class="m">{l}</text>')
    if i < 4:
        add(f'<path d="M{x + CW + 1},{y0 + IH / 2} H{CX[i + 1] - 1}" class="arrow"/>')

ly = CY + CH + 26
leg = [("line", "#f2b01e", "검출기 상자"), ("dot", "#1f6fd1", "안쪽 점 (+)"), ("ring", "#1b1d21", "바깥 점 (−)"), ("line", "#ffffff", "SAM 후보"),
       ("line", "#d63fd6", "후보 합의"), ("line", "#e23b2f", "최종 화면 영역")]
lx = 40
for kind, col, label in leg:
    if kind == "line":
        stroke = "#9aa3ad" if col == "#ffffff" else col
        add(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 20}" y2="{ly - 4}" stroke="{stroke}" stroke-width="3"/>')
    elif kind == "dot":
        add(f'<circle cx="{lx + 8}" cy="{ly - 4}" r="6" fill="{col}"/>')
    else:
        add(f'<circle cx="{lx + 8}" cy="{ly - 4}" r="6" fill="#fff" stroke="{col}" stroke-width="2"/>')
    add(f'<text x="{lx + 26}" y="{ly}" class="s">{label}</text>')
    lx += 26 + 12 * len(label) + 30
add(f'<text x="1560" y="{ly}" class="s" text-anchor="end">꼭짓점을 직접 끌어 고치면 ⑤만 다시 실행 (테스트 페이지)</text>')

# ---------- chart: HD95 by group for the two methods ----------
PY = 610
add(f'<rect x="40" y="{PY}" width="900" height="335" rx="16" class="panel"/>')
add(f'<text x="60" y="{PY + 32}" class="h">화면 경계 오차 HD95 (화면 대각선 대비 %, 낮을수록 좋음)</text>')
add(f'<text x="60" y="{PY + 54}" class="m">정답 경계와 찾은 경계 사이 거리의 95번째 백분위 · 52장, 정답은 손으로 표시</text>')
groups, methods = nums["groups"], nums["methods"]
COL = {"VLM+SAM (gemma4-12b)": "#c4652a", "검출기+SAM": "#5b8fc7"}
lgx = 560
for k, (mname, col) in enumerate(COL.items()):
    add(f'<rect x="{lgx + k * 190}" y="{PY + 22}" width="12" height="12" rx="2" fill="{col}"/><text x="{lgx + k * 190 + 18}" y="{PY + 33}" class="s">{mname}</text>')
X0, R0, RP = 250, PY + 82, 40
vmax = max(max(m["hd"]) for m in methods.values())
XW = 600 / max(10.0, np.ceil(vmax / 5) * 5)
scale_max = 600 / XW
for k in range(0, int(scale_max) + 1, 5):
    gx = X0 + XW * k
    add(f'<line x1="{gx:.1f}" y1="{R0 - 8}" x2="{gx:.1f}" y2="{R0 + RP * len(groups) - 8}" stroke="{"#c9ced6" if k == 0 else "#eceef1"}"/>')
    add(f'<text x="{gx:.1f}" y="{R0 + RP * len(groups) + 8}" class="s" text-anchor="middle">{k}</text>')
for gi, (label, n) in enumerate(groups):
    y = R0 + gi * RP
    add(f'<text x="{X0 - 12}" y="{y + 17}" class="b" text-anchor="end">{label} ({n}장)</text>')
    for mi, (mname, col) in enumerate(COL.items()):
        v = methods[mname]["hd"][gi]
        add(f'<rect x="{X0}" y="{y + mi * 14}" width="{max(2, XW * v):.1f}" height="12" rx="2" fill="{col}"/>')
        add(f'<text x="{X0 + XW * v + 6:.1f}" y="{y + mi * 14 + 10}" class="{"v" if mi else "s"}">{v:.1f}</text>')
add(f'<text x="60" y="{PY + 322}" class="m">근접 6장 = 키오스크 앞에서 직접 찍은 사진 (3장은 주문 확인 팝업이 뜬 상태)</text>')

# ---------- right panel ----------
RX = 960
d, v = methods["검출기+SAM"], methods["VLM+SAM (gemma4-12b)"]
add(f'<rect x="{RX}" y="{PY}" width="600" height="335" rx="16" class="panel"/>')
add(f'<text x="{RX + 20}" y="{PY + 32}" class="h">이전 방식(VLM+SAM) 대비 · 남은 문제</text>')
add(f'<rect x="{RX + 20}" y="{PY + 48}" width="270" height="84" rx="10" fill="#f3f5f8"/>')
add(f'<text x="{RX + 36}" y="{PY + 90}" class="big">{v["sec"]:.1f}초 → {d["sec"]:.1f}초</text>')
add(f'<text x="{RX + 36}" y="{PY + 116}" class="m">사진 한 장 처리 시간 (중앙값)</text>')
add(f'<rect x="{RX + 310}" y="{PY + 48}" width="270" height="84" rx="10" fill="#f3f5f8"/>')
add(f'<text x="{RX + 326}" y="{PY + 90}" class="big">{v["cuts"]} → {d["cuts"]}장</text>')
add(f'<text x="{RX + 326}" y="{PY + 116}" class="m">화면이 2% 넘게 잘린 사진 (52장 중)</text>')
probs = [("팝업 + 강한 창문 반사가 겹치면 경계를 못 찾음", "변 하나만 맞춰져 비스듬하게 펴짐"),
         ("베젤 바깥이 어두운 배경이면 팝업 옆 여백이 잘림", "베젤 바깥 테두리를 찾지 못하는 경우"),
         ("베젤을 조금 더 포함하는 쪽으로 치우침", "근접 사진 평균 넘침 약 4.7%"),
         ("평가용 근접 사진이 아직 6장뿐", "→ 사진을 모아 작은 분할 모델 미세조정 예정")]
yy = PY + 164
for a, b in probs:
    add(f'<circle cx="{RX + 26}" cy="{yy - 5}" r="3" fill="#6b737e"/><text x="{RX + 36}" y="{yy}" class="b">{a}</text>')
    add(f'<text x="{RX + 36}" y="{yy + 19}" class="m">{b}</text>')
    yy += 42
add('<text x="40" y="975" class="s">예시 사진: Wikimedia Commons — Asacyan, 「Korail ar seoul sta 02」 (CC BY-SA 3.0), 화면 주변을 잘라 근접 구도로 사용 · 2026-11-18 기준</text>')
add("</svg>")
open("docs/rectify_pipeline.svg", "w").write("\n".join(o))
print("ok", len("\n".join(o)))
