"""키포인트 → **얼굴 상자** · **목선** · **가로 기준점**. 이 실험의 심장.

앱(`DAENGS_APP`)에는 이미 이 둘을 받는 자리가 뚫려 있다. 새로 만드는 게 아니라
지금 사람이 손으로 채우고 있는 것을 기계가 채우게 하는 것이다.

    Cutout.of(photo, box)          box  = 정규화 [x, y, w, h]   ← face_box()
    Cutout.Result.Cut.neck         neck = 상자 안 0~1 세로 위치  ← neck_t()

`neck` 쪽은 사연이 있다. 앱은 지금 **실루엣의 가로 폭이 그리는 골짜기**로 목을
짐작하는데, 커밋 f5c2b1d 가 적어 뒀듯 털 많은 개(포메라니안·골든리트리버)는 목이
안 좁아져서 무너진다. dog-pose 에 `throat` 키포인트가 있길래 그걸 그대로 쓰려고
이 데이터셋을 골랐는데, **막상 세어 보니 8,476장 전부에서 안 찍혀 있었다**
(`keypoints.py` 의 경고). 그래서 목은 턱과 귀밑에서 **추정한다** — 그래도 실루엣
골짜기보다는 낫다. 털은 변해도 두개골은 안 변하기 때문이다.

## 왜 픽셀로 계산하나

라벨은 정규화 좌표라 x 와 y 의 한 칸이 서로 다른 길이다. 정규화 상태로 "머리 크기의
25% 만큼 키운다" 를 하면 **가로로 긴 사진에서 상자가 납작해진다.** 앱의
`GuideFrame.kt` 가 `h = w * aspect` 로 원본 픽셀 기준 정사각을 만드는 것과 같은
이유다. 그래서 여기서는 들어오자마자 픽셀로 바꾸고, 나갈 때 다시 정규화한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import dogpose as kp

#: 얼굴 키포인트가 이보다 적게 보이면 상자를 안 만든다.
#:
#: **못 만드는 것이 아무 상자나 만드는 것보다 낫다.** 앱은 이때
#: `GuideFrameScreen` 으로 물러서면 되고, 그건 지금도 있는 길이다. 하나만 보고
#: 상자를 만들면 그 하나가 코일 때와 귀끝일 때 결과가 전혀 달라진다.
MIN_FACE_KPS = 3

#: 목선을 못 찾았을 때 쓰는 값. **앱의 `Cutout.NECK_FALLBACK` 과 같은 숫자다.**
#: 두 곳이 갈라지면 파이썬에서 본 결과가 앱에서 재현이 안 된다.
NECK_FALLBACK = 0.74

#: 목선을 이 범위 밖으로는 안 보낸다. 라벨이 뒤집힌 이미지에서 목이 이마 위로
#: 올라오면 얼굴이 통째로 녹는다 — 검증 중에 실제로 그런 라벨을 봤다.
NECK_MIN, NECK_MAX = 0.35, 1.0

#: 세로가 모자랄 때 **위로 주는 몫.** 나머지는 아래로 간다.
#:
#: 정사각으로 만들 때 위아래에 똑같이 나눠 주면 상자가 가슴으로 내려간다 —
#: 검증에서 오염 범인 1·2위가 앞다리 팔꿈치(20.1% · 14.8%)로 나온 것이 그것이다.
#: 머리는 **위에 있고 가슴은 아래에 있으므로** 모자란 세로는 위로 주는 것이 맞다.
SQUARE_UP = 0.75

#: 상자 경계에 딱 걸친 점을 밖으로 보지 않기 위한 여유.
#:
#: 마진 0 일 때 상자는 정의상 얼굴 점들의 경계상자라 점이 변 위에 정확히 놓인다.
#: 그런데 라벨이 float32 라 정규화 → 픽셀 → 정규화를 오가면 1e-8 쯤 어긋나서,
#: **재현율이 28.6% 로 찍혔다.** 상자가 틀린 게 아니라 자가 틀렸던 것이다.
EDGE_EPS = 1e-6


@dataclass(frozen=True)
class Margins:
    """얼굴 키포인트 상자를 머리 크기 대비 얼마나 키울지.

    **위아래가 다르다.** 키포인트는 이마·정수리·뒤통수를 안 찍는다 — 제일 위가
    `ear_base`/`ear_tip` 이라 그대로 쓰면 이마가 잘린다. 아래는 `chin` 이 이미
    턱 끝이라 조금만 있으면 된다.

    기본값은 짐작이 아니라 `tools/verify_face_box.py --sweep` 이 val 1,703장(라벨이
    어긋난 72장 제외)에서 27칸을 훑어 고른 값이다. 통과선(구멍 안 오염률 20% 이하)을
    넘는 칸 중 **이마 여유(top)를 가장 크게 남기는** 것을 골랐다 — 구멍 모서리의
    앞발은 안 보이지만 잘린 이마는 보이기 때문이다.

        top  side bottom |  재현율  오염률  넓이비
        0.30 0.15 0.35   |  1.000   0.179  0.454
        0.45 0.15 0.35   |  1.000   0.197  0.540   ← 이것
        0.45 0.25 0.55   |  1.000   0.239  0.681
        0.60 0.35 0.75   |  1.000   0.280  0.951

    `side` 는 거의 영향이 없다 (0.15 → 0.35 에서 넓이비 0.540 → 0.593). 정사각으로
    만드는 단계가 가로를 어차피 채우기 때문이다.
    """

    top: float = 0.45
    side: float = 0.15
    #: 아래는 **턱에서 끊지 않는다.** 딱 끊으면 누끼 아래가 직선이 되고 흰 테두리
    #: 띠가 가로로 그어져서, 카드에 얹으면 "사진을 오려 붙인 것" 으로 보인다
    #: (PR #19 커밋 f229473 이 실기기에서 본 것). 목을 조금 담아 두고 [neck_t]
    #: 아래를 녹이는 쪽이 낫다. 다만 크게 잡을수록 가슴과 앞다리가 구멍에 들어온다 —
    #: 0.75 로 두면 오염률이 19.7% 에서 25.9% 로 뛴다.
    bottom: float = 0.35


DEFAULT_MARGINS = Margins()


@dataclass(frozen=True)
class Box:
    """정규화 `[x, y, w, h]`. **앱의 `Cutout.of(photo, box)` 와 같은 모양이다.**"""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    def to_pixels(self, iw: int, ih: int) -> tuple[int, int, int, int]:
        """`(left, top, right, bottom)` 픽셀. Pillow 의 `crop` 에 그대로 넣는다."""
        return (
            int(round(self.x * iw)),
            int(round(self.y * ih)),
            int(round(self.x1 * iw)),
            int(round(self.y1 * ih)),
        )

    def as_list(self) -> list[float]:
        """앱에 넘길 `FloatArray` 와 같은 순서."""
        return [self.x, self.y, self.w, self.h]

    def contains(self, x: float, y: float, eps: float = EDGE_EPS) -> bool:
        """변 위에 놓인 점은 **안에 든 것으로 본다** ([EDGE_EPS] 참고)."""
        return (
            self.x - eps <= x <= self.x1 + eps
            and self.y - eps <= y <= self.y1 + eps
        )


def visible(kps: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    """`idx` 중 실제로 찍힌 것들의 `(x, y)`. 정규화 그대로.

    **`v` 를 봐야 한다.** 안 찍힌 키포인트는 `x = y = 0` 으로 저장돼 있어서, `v` 를
    안 보면 전부 왼쪽 위 모서리로 몰려 상자가 그쪽으로 늘어난다.
    """
    sel = kps[list(idx)]
    return sel[sel[:, 2] > 0][:, :2]


def face_box(
    kps: np.ndarray,
    iw: int,
    ih: int,
    margins: Margins = DEFAULT_MARGINS,
) -> Box | None:
    """얼굴 키포인트에서 **정사각형** 얼굴 상자를 만든다. 못 만들면 `None`.

    @param kps (24, 3) 정규화 배열. `[x, y, v]`
    @param iw,ih 원본 픽셀 크기. 정사각을 픽셀 기준으로 만들기 위해 필요하다
    """
    pts = visible(kps, kp.FACE)
    if len(pts) < MIN_FACE_KPS:
        return None

    # 픽셀로 옮긴다. 이 아래는 전부 픽셀이다.
    px = pts[:, 0] * iw
    py = pts[:, 1] * ih
    x0, x1 = float(px.min()), float(px.max())
    y0, y1 = float(py.min()), float(py.max())

    # 머리 크기의 자. 짧은 쪽을 쓰면 옆모습(코~귀만 보이는)에서 자가 0 에 가까워져
    # 마진이 사라진다. **긴 쪽을 쓴다.**
    head = max(x1 - x0, y1 - y0)
    if head <= 0:
        return None

    x0 -= margins.side * head
    x1 += margins.side * head
    y0 -= margins.top * head
    y1 += margins.bottom * head

    # 정사각으로. 앱의 카드 구멍이 둘 다 원이라 가로세로가 갈리면 얼굴이 눌린다.
    #
    # **모자란 축을 채우는 방향이 축마다 다르다.** 가로는 좌우 대칭으로 나누지만
    # (얼굴이 좌우 대칭이다), 세로는 [SQUARE_UP] 만큼 위로 몰아준다 — 반씩 나누면
    # 상자가 가슴으로 내려가 앞다리를 먹는다.
    w, h = x1 - x0, y1 - y0
    if w > h:
        d = w - h
        y0 -= d * SQUARE_UP
        y1 += d * (1.0 - SQUARE_UP)
    else:
        d = h - w
        x0 -= d / 2
        x1 += d / 2

    side = x1 - x0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    # 사진 밖으로 나간 만큼 **중심을 민다.** 잘라내면 정사각이 깨진다.
    #
    # 그래도 안 들어가면(머리가 사진 짧은 변보다 크다) **정사각을 포기한다.**
    # 여기서 변을 줄이면 귀나 코가 상자 밖으로 나가는데, 카드 구멍이 원이라
    # 조금 눌린 얼굴보다 잘린 귀가 훨씬 나쁘다.
    if side <= min(iw, ih):
        left = min(max(cx - side / 2, 0.0), iw - side)
        top = min(max(cy - side / 2, 0.0), ih - side)
        return Box(left / iw, top / ih, side / iw, side / ih)

    left, right = max(x0, 0.0), min(x1, float(iw))
    top, bottom = max(y0, 0.0), min(y1, float(ih))
    return Box(left / iw, top / ih, (right - left) / iw, (bottom - top) / ih)


#: `chin` 에서 아래로 얼마나 내려가야 목인가. **머리 길이(귀밑~턱) 대비**다.
#:
#: ⚠️ 원래는 `throat`(23) 을 그대로 쓰려 했다. 그게 이 데이터셋을 고른 이유였다.
#: 그런데 **`throat` 은 train·val 8,476장 전부에서 한 번도 안 찍혀 있다** —
#: yaml 이 이름만 선언하고 라벨은 없다 (`keypoints.py` 의 경고 참고).
#:
#: 그래서 목은 **재는 것이 아니라 추정하는 것**이 됐다. 다만 앱이 지금 쓰는
#: 실루엣 폭 골짜기보다는 낫다 — 실루엣은 털에 좌우되지만 귀밑~턱은 두개골이라
#: 털이 아무리 많아도 안 변한다. 그게 포메라니안·골든리트리버에서 앱이 무너진
#: 이유였다.
#:
#: 0.25 는 컨택트시트를 보고 정한 값이다. **실측이 아니라 눈으로 고른 것**이라,
#: 앱에 옮길 때 실사 사진에서 다시 봐야 한다. 크게 잡으면 목선이 상자 밖으로
#: 나가 [NECK_MAX] 에 걸려 버려서 페이드가 아무 일도 안 하게 된다 — 처음에 0.45
#: 로 뒀다가 목선 중앙값이 0.98 로 찍혔다.
CHIN_DROP = 0.25


def neck_t(kps: np.ndarray, box: Box, ih: int) -> float:
    """[box] 안에서 목이 있는 세로 위치(0~1). 앱의 `Result.Cut.neck` 과 같은 뜻.

    세 갈래로 내려간다.

    1. `throat` 이 찍혀 있으면 그 자리 — **이 데이터셋에서는 절대 안 걸린다.**
       다른 데이터셋(AnimalPose 등)을 붙일 때를 위해 남겨 둔 길이다
    2. `chin` 과 귀밑이 있으면 턱 아래로 머리 길이의 [CHIN_DROP] 만큼
    3. 아무것도 없으면 [NECK_FALLBACK] — 앱이 지금 쓰는 값
    """
    y0 = box.y * ih
    span = box.h * ih
    if span <= 0:
        return NECK_FALLBACK

    def to_t(y_norm: float) -> float:
        return float(np.clip((y_norm * ih - y0) / span, NECK_MIN, NECK_MAX))

    throat = kps[kp.THROAT]
    if throat[2] > 0:
        return to_t(float(throat[1]))

    chin = kps[kp.CHIN]
    ears = visible(kps, kp.EAR_BASES)
    if chin[2] > 0 and len(ears) > 0:
        # 머리 길이 = 귀밑에서 턱까지. 코까지가 아니다 — 주둥이가 긴 개(콜리)와
        # 짧은 개(퍼그)에서 코~턱 길이는 몇 배씩 차이 나지만 귀밑~턱은 덜하다.
        head_len = abs(float(chin[1]) - float(ears[:, 1].mean()))
        return to_t(float(chin[1]) + CHIN_DROP * head_len)

    return NECK_FALLBACK


#: 가로 기준점을 코 쪽으로 당기는 몫. 1.0 이면 코, 0.0 이면 귀밑 한가운데.
#:
#: **정면 사진에서는 아무 차이가 없다** — 코와 귀밑 한가운데가 거의 같은 x 다.
#: 옆모습에서만 작동하고, 거기서는 반드시 필요하다: 먼 쪽 귀밑이 뒤통수에 있어서
#: 단순 평균을 쓰면 기준점이 머리 뒤로 끌려간다. 페키니즈 옆모습에서 코가 x=147
#: 인데 먼 귀밑이 x=57 이라 평균이 119 로 나왔고, 카드 구멍에 **얼굴 대신 털**이
#: 들어갔다.
ANCHOR_NOSE_WEIGHT = 0.7


def face_anchor_x(kps: np.ndarray, iw: int) -> float | None:
    """구멍에 맞출 **가로 기준점**(원본 픽셀). 얼굴 키포인트가 모자라면 `None`.

    코와 귀밑 한가운데를 [ANCHOR_NOSE_WEIGHT] 로 섞는다. 눈이 라벨에 없어서
    "두 눈 사이" 를 못 쓰는 대신 찾은 방법이다.
    """
    nose = kps[kp.NOSE]
    ears = visible(kps, kp.EAR_BASES)
    if nose[2] > 0 and len(ears) > 0:
        w = ANCHOR_NOSE_WEIGHT
        return (w * float(nose[0]) + (1 - w) * float(ears[:, 0].mean())) * iw

    # 코나 귀밑 중 하나가 없으면 있는 것들의 평균으로 물러선다.
    pts = visible(kps, kp.FACE)
    return float(pts[:, 0].mean()) * iw if len(pts) else None


def body_box(kps: np.ndarray) -> Box | None:
    """찍힌 키포인트 전부의 상자. 정규화.

    **라벨 파일의 `cx cy w h` 를 안 쓴다.** val 300장을 재 봤더니 6.7% 에서 키포인트가
    그 상자 밖으로 나가고, 일부(`n02085620_1152` 등)는 상자가 좌우로 뒤집힌 것처럼
    어긋나 있었다. 넓이비를 재는 자로 쓰기에는 못 믿는다.
    """
    pts = kps[kps[:, 2] > 0][:, :2]
    if len(pts) < 2:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return Box(float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def in_hole(box: Box, x: float, y: float) -> bool:
    """[box] 에 내접하는 **타원** 안인가.

    카드 구멍이 타원이라 (`CardSlots.kt` 의 `Hole`), 상자 네 모서리에 있는 것은
    카드에 아예 안 보인다. 사각형으로 오염을 재면 **보이지도 않는 앞발 때문에
    마진을 줄이게 되고, 그 대가로 이마가 잘린다.**
    """
    if box.w <= 0 or box.h <= 0:
        return False
    nx = (x - (box.x + box.w / 2)) / (box.w / 2)
    ny = (y - (box.y + box.h / 2)) / (box.h / 2)
    return nx * nx + ny * ny <= 1.0


def body_hits(
    kps: np.ndarray,
    box: Box,
    above: float | None = None,
    hole: bool = False,
) -> list[int]:
    """[box] 안에 든 몸통 키포인트의 인덱스.

    @param above 주면 **이 세로 위치(상자 안 0~1) 위쪽만** 센다.

    목선 아래는 어차피 `fadedBelow` 가 녹인다. 그래서 "상자 안에 앞발이 있다" 는
    것만으로 오염이라고 하면 지표가 과하게 나쁘게 나온다 — 실제로 기본 마진에서
    30.4% 가 찍혔는데 범인 1·2위가 앞다리 팔꿈치였고, 그것들은 대부분 턱 아래라
    카드에서는 보이지도 않는다. **문제는 목선 위에 남는 몸이다.**
    """
    limit = None if above is None else box.y + box.h * above
    out = []
    for i in kp.BODY:
        x, y, v = kps[i]
        if v <= 0 or not box.contains(float(x), float(y)):
            continue
        if limit is not None and float(y) >= limit:
            continue
        if hole and not in_hole(box, float(x), float(y)):
            continue
        out.append(i)
    return out
