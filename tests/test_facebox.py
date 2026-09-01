"""얼굴 상자 · 목선 · 가로 기준점의 좌표 계산.

**이 테스트가 잡는 것과 못 잡는 것을 갈라 둔다.** 여기서 잡는 것은 좌표다. 카드가
예쁜지, 얼굴이 구멍에 잘 앉았는지는 **못 잡는다** — 그건 `out/face-cards/` 를 눈으로
보는 수밖에 없다 (그림 판단은 사람이 한다, 앱 협업규칙 1절).
"""

import numpy as np
import pytest

from src import dogpose as kp
from src import facebox as geo


def make_kps(**named: tuple[float, float]) -> np.ndarray:
    """이름으로 키포인트를 놓는다. 안 준 것은 `v = 0` 이라 안 찍힌 것이 된다."""
    a = np.zeros((kp.N_KEYPOINTS, 3), dtype=np.float32)
    for name, (x, y) in named.items():
        a[kp.INDEX[name]] = (x, y, 2.0)
    return a


# -- 안 찍힌 키포인트 -------------------------------------------------------


def test_안_찍힌_점은_원점으로_안_끌려간다():
    """`v` 를 안 보면 (0, 0) 에 몰린 점들이 상자를 왼쪽 위로 늘린다."""
    kps = make_kps(nose=(0.6, 0.5), chin=(0.6, 0.6), left_ear_base=(0.5, 0.4))
    box = geo.face_box(kps, 400, 400)
    assert box is not None
    assert box.x > 0.1, "상자가 왼쪽 위 모서리로 끌려갔다"


def test_얼굴_키포인트가_모자라면_상자를_안_만든다():
    """물러설 자리를 남긴다 — 앱에서는 사람이 직접 맞춘다."""
    kps = make_kps(nose=(0.5, 0.5), chin=(0.5, 0.6))
    assert geo.face_box(kps, 400, 400) is None


def test_눈과_throat_은_이_데이터셋에_없다():
    """설계가 없는 키포인트에 기대지 않는지 못박아 둔다."""
    for name in ("left_eye", "right_eye", "withers", "throat"):
        assert kp.INDEX[name] in kp.UNLABELED
        assert kp.INDEX[name] not in kp.FACE


# -- 상자 모양 --------------------------------------------------------------


def test_상자는_원본_픽셀_기준_정사각이다():
    """정규화 기준으로 만들면 가로로 긴 사진에서 납작해진다."""
    kps = make_kps(nose=(0.5, 0.5), chin=(0.5, 0.6), left_ear_base=(0.45, 0.4),
                   right_ear_base=(0.55, 0.4))
    iw, ih = 800, 400
    box = geo.face_box(kps, iw, ih)
    assert box is not None
    assert box.w * iw == pytest.approx(box.h * ih, rel=1e-3)


def test_상자는_사진_밖으로_안_나간다():
    kps = make_kps(nose=(0.97, 0.97), chin=(0.99, 0.99), left_ear_base=(0.93, 0.93))
    box = geo.face_box(kps, 500, 500)
    assert box is not None
    assert box.x >= 0 and box.y >= 0
    assert box.x1 <= 1.0 + 1e-6 and box.y1 <= 1.0 + 1e-6


def test_모자란_세로는_위로_더_준다():
    """반씩 나누면 상자가 가슴으로 내려가 앞다리를 먹는다."""
    # 가로로 넓은 얼굴 — 세로를 채워야 정사각이 된다.
    kps = make_kps(left_ear_tip=(0.30, 0.50), right_ear_tip=(0.70, 0.50),
                   nose=(0.50, 0.55), chin=(0.50, 0.58))
    box = geo.face_box(kps, 600, 600, geo.Margins(top=0.0, side=0.0, bottom=0.0))
    assert box is not None
    kp_top, kp_bottom = 0.50, 0.58
    above = kp_top - box.y
    below = box.y1 - kp_bottom
    assert above > below, "세로 여유가 위아래 반씩 나뉘었다"


def test_마진_0_이면_얼굴_점이_전부_상자_안에_있다():
    """변에 딱 걸친 점을 밖으로 세면 재현율이 28.6% 로 찍힌다 (EDGE_EPS)."""
    kps = make_kps(nose=(0.6, 0.55), chin=(0.6, 0.62), left_ear_base=(0.52, 0.44),
                   right_ear_base=(0.68, 0.45), left_ear_tip=(0.50, 0.38))
    box = geo.face_box(kps, 640, 480, geo.Margins(0.0, 0.0, 0.0))
    assert box is not None
    for x, y in geo.visible(kps, kp.FACE):
        assert box.contains(float(x), float(y))


# -- 목선 ------------------------------------------------------------------


def test_목선은_턱보다_아래다():
    kps = make_kps(nose=(0.5, 0.50), chin=(0.5, 0.60), left_ear_base=(0.45, 0.40),
                   right_ear_base=(0.55, 0.40))
    ih = 500
    box = geo.face_box(kps, 500, ih)
    assert box is not None
    t = geo.neck_t(kps, box, ih)
    chin_t = (0.60 - box.y) / box.h
    assert t > chin_t


def test_목선을_못_찾으면_앱과_같은_기본값():
    """두 곳이 갈라지면 파이썬에서 본 결과가 앱에서 재현이 안 된다."""
    kps = make_kps(nose=(0.5, 0.5), left_ear_tip=(0.4, 0.4), right_ear_tip=(0.6, 0.4))
    box = geo.face_box(kps, 400, 400)
    assert box is not None
    assert geo.neck_t(kps, box, 400) == geo.NECK_FALLBACK


# -- 가로 기준점 ------------------------------------------------------------


def test_옆모습에서_기준점이_뒤통수로_안_끌려간다():
    """단순 평균을 쓰면 먼 쪽 귀밑에 끌려가 구멍에 얼굴 대신 털이 들어간다."""
    iw = 400
    kps = make_kps(nose=(0.80, 0.55), left_ear_base=(0.60, 0.40),
                   right_ear_base=(0.30, 0.45), chin=(0.80, 0.62))
    anchor = geo.face_anchor_x(kps, iw)
    plain_mean = float(geo.visible(kps, kp.FACE)[:, 0].mean()) * iw
    assert anchor is not None
    assert anchor > plain_mean, "기준점이 단순 평균보다 코 쪽으로 안 왔다"


def test_정면에서는_기준점이_평균과_거의_같다():
    """가중치는 옆모습에서만 작동해야 한다."""
    iw = 400
    kps = make_kps(nose=(0.50, 0.55), left_ear_base=(0.44, 0.40),
                   right_ear_base=(0.56, 0.40), chin=(0.50, 0.62))
    anchor = geo.face_anchor_x(kps, iw)
    assert anchor == pytest.approx(0.50 * iw, abs=1.0)
