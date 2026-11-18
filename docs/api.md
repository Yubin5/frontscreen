# 화면 펴기 API

> 2026-11-18 기준 · 서버 코드: `api_server.py`

사진을 보내면, 사진 속 디스플레이 화면(키오스크 화면 등)을 찾아 정면에서 본 이미지(JPEG)로 펴서 돌려준다.

## 사용법

```
POST /flatten
본문: 사진 파일 그대로 (JPEG / PNG / WebP, 20MB 이하)
응답: 펴진 화면 JPEG
```

```bash
curl -s -X POST --data-binary @photo.jpg http://127.0.0.1:8793/flatten -o screen.jpg
```

```python
import requests
r = requests.post("http://127.0.0.1:8793/flatten", data=open("photo.jpg", "rb").read(), timeout=30)
r.raise_for_status()
open("screen.jpg", "wb").write(r.content)
```

- **처리 시간:** 보통 1초 안팎이다. 결과가 미심쩍어 다시 시도하는 사진은 1.7초쯤 걸린다. 5분 넘게 요청이 없었다면 첫 요청은 모델을 다시 올리느라 몇 초 더 걸린다.
- **크기:** 펴진 이미지의 긴 변은 1600px이다. 바꾸려면 `?max_side=1200`처럼 준다(256~4096).

## 주소와 인증

- **기본 주소:** `http://127.0.0.1:8793`이다. 서버 안에서만 열린다.
- **다른 기기에서 부르기:** `API_HOSTS`에 바인드 주소를 추가한다(예: `API_HOSTS=127.0.0.1,<사설망 IP>`). 이 경우 `secrets/api_key` 파일에 키를 넣어 두면, 서버 밖에서 오는 요청은 `Authorization: Bearer <키>` 헤더가 있어야 한다.
- **인터넷 공개:** 직접 열지 않는 것을 권한다.

## 응답 헤더 (필요할 때만 보면 된다)

| 헤더 | 내용 |
|---|---|
| `X-Screen-Check` | `ok` 또는 `doubtful`. `doubtful`이면 화면을 잘못 잡았을 수 있으니, 사용자에게 결과를 한 번 확인받는 게 좋다 |
| `X-Screen-Corners` | 원본 사진에서 찾은 화면 꼭짓점. 왼쪽위·오른쪽위·오른쪽아래·왼쪽아래 순서이고, 사진 너비·높이 대비 비율이다. 예: `0.1802,0.0633;0.8374,0.0533;0.8291,0.8795;0.2359,0.8866` |
| `X-Process-Ms` | 서버 처리 시간(ms) |

## 오류

오류일 때는 `{"error": "설명"}` JSON이 온다.

| 코드 | 언제 |
|---|---|
| 400 | 본문이 비었거나 이미지로 읽을 수 없을 때 |
| 401 | 서버 밖에서 부르면서 API 키가 없거나 틀렸을 때 |
| 413 | 20MB를 넘을 때 |
| 422 | 사진에서 화면을 못 찾았을 때 |
| 503 | 화면 검출 서버(SAM·검출기)가 꺼져 있을 때 |

## 운영

```bash
docker build -t frontscreen-api .          # API 이미지 (처음 한 번)
docker build -t frontscreen-sam sam/      # 검출 서버 이미지, CPU용 (처음 한 번)
./run_api.sh          # 시작 (검출 서버가 꺼져 있으면 먼저 켠다)
./run_api.sh status   # 상태 확인 (GET /health → {"ok": true})
./run_api.sh stop     # API만 내린다
```

- **컨테이너:** `frontscreen-api`(이 API)와 `frontscreen-sam`(SAM + Grounding DINO, `127.0.0.1:8791`)이다. 이미지·이름·주소는 `local.env`로 바꿀 수 있다(`run_api.sh` 첫머리 참고).
- **GPU:** 기본은 CPU다(사진 한 장에 수 초). Intel Arc에서는 XPU용 PyTorch가 든 이미지로 `SAM_IMAGE`, `SAM_DEVICE=xpu`를 주면 약 1초다.
- **코드 반영:** 두 컨테이너 모두 이 폴더를 마운트해서 쓰므로, 코드를 고친 뒤 컨테이너만 재시작하면 된다.
- **동시 요청:** 하나씩 차례로 처리한다.

## 동작 원리와 한계

동작 원리는 [rectify_pipeline.md](rectify_pipeline.md)에 있다. 요약하면 다음 순서다.
1. 검출기(Grounding DINO)로 화면 상자를 찾는다.
2. SAM으로 화면 영역을 잡는다.
3. 근접 사진이면 베젤 경계에 맞춘다.
4. 직각·기울기 검사를 하고, 걸리면 보정하거나 다른 후보로 다시 시도한다.
5. 원근에서 실제 가로세로비를 복원해 편다.

- **오차 경향:** 화면을 자르기보다 베젤을 조금 더 포함하는 쪽으로 틀린다.
- **화면이 여러 개인 사진:** 하나만 돌려준다.
- **광각 사진, 위에서 크게 내려다본 사진:** 결과가 맞아도 `doubtful`이 나올 수 있다.
