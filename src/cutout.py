"""누끼 — 사진에서 강아지만 남긴다.

rembg 는 무겁고(onnxruntime + 가중치) 첫 실행에 모델을 받아야 해서 선택 의존성으로
갈라 두었다 (`uv sync --extra cutout`). `skin-screening` 의 `--extra model` 과 같은
방식이다. 없으면 **곱게 실패한다** — 503 성격이지 500 이 아니다.

알파가 이미 있는 PNG 를 주면 rembg 없이도 나머지 파이프라인이 그대로 돈다.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from src.compose import trim_alpha

# 이 아래 알파는 누끼 가장자리의 먼지로 본다. 남겨 두면 bbox 가 사진 전체가 된다.
ALPHA_DUST = 8


class FaceCutoutUnavailable(RuntimeError):
    """opencv 가 없다. `uv sync --extra facebox` 로 켠다 (CA-005 와 같은 방식)."""


def _cv2():
    """opencv 를 쓰는 자리에서만 부른다. 기본 설치에는 없다."""
    try:
        import cv2
    except Exception as exc:
        raise FaceCutoutUnavailable(
            "opencv 가 없다. `uv sync --extra facebox` 를 하면 얼굴 누끼가 돈다"
        ) from exc
    return cv2


class CutoutUnavailable(RuntimeError):
    """rembg 가 없거나 모델을 못 받았다. 설비가 없는 것이지 코드가 틀린 게 아니다."""


@dataclass(frozen=True)
class Cutout:
    image: Image.Image                 # 알파 있는 강아지, 여백은 잘려 있다
    bbox: tuple[int, int, int, int]    # 원본 사진에서의 자리 (left, top, right, bottom)
    source_size: tuple[int, int]
    used_rembg: bool

    @property
    def coverage(self) -> float:
        """사진에서 강아지가 차지하는 넓이 비율. 누끼가 통째로 실패했는지 보는 눈."""
        w, h = self.source_size
        left, top, right, bottom = self.bbox
        return ((right - left) * (bottom - top)) / (w * h) if w and h else 0.0


def available() -> bool:
    try:
        import rembg  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _session(model: str):
    try:
        from rembg import new_session
    except Exception as exc:                     # ImportError 만이 아니다 — onnxruntime 도 여기서 터진다
        raise CutoutUnavailable(
            "rembg 가 없다. `uv sync --extra cutout` 을 하거나, 알파가 이미 있는 PNG 를 넣는다"
        ) from exc
    try:
        return new_session(model)
    except Exception as exc:                     # 가중치를 받는 길이 막혔을 때
        raise CutoutUnavailable(f"누끼 모델({model})을 준비하지 못했다: {exc}") from exc


def has_alpha(img: Image.Image) -> bool:
    """쓸 만한 알파가 이미 있는가. 전부 불투명하면 없는 것으로 본다."""
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return False
    alpha = img.convert("RGBA").getchannel("A")
    return alpha.getextrema()[0] < 255


def _clean(rgba: Image.Image) -> Image.Image:
    alpha = rgba.getchannel("A").point(lambda v: 0 if v < ALPHA_DUST else v)
    out = rgba.copy()
    out.putalpha(alpha)
    return out


def cutout(img: Image.Image, *, model: str = "u2net", reuse_alpha: bool = True) -> Cutout:
    """강아지만 남긴 RGBA 를 돌려준다.

    reuse_alpha 면 이미 알파가 있는 그림에는 rembg 를 부르지 않는다 — 누끼가 이미
    된 PNG 로 파이프라인을 돌릴 수 있어야 하고, 그것이 이 단계 없이도 나머지가
    도는 유일한 길이다.
    """
    source_size = img.size

    if reuse_alpha and has_alpha(img):
        rgba, used = _clean(img.convert("RGBA")), False
    else:
        session = _session(model)
        try:
            from rembg import remove

            rgba = remove(img.convert("RGB"), session=session).convert("RGBA")
        except Exception as exc:
            raise CutoutUnavailable(f"누끼를 뜨지 못했다: {exc}") from exc
        rgba, used = _clean(rgba), True

    box = rgba.getchannel("A").point(lambda v: 255 if v >= ALPHA_DUST else 0).getbbox()
    if box is None:
        raise CutoutUnavailable("누끼 결과가 통째로 비었다 — 강아지를 못 찾았다")

    return Cutout(image=trim_alpha(rgba), bbox=box, source_size=source_size, used_rembg=used)


# ---------------------------------------------------------------------------
# 얼굴 구멍용 누끼 — 가중치 없이 도는 갈래
#
# 위쪽 [cutout] 은 rembg 로 **강아지 전체**를 딴다. 아래는 **얼굴 상자 안에서만**
# 배경을 지우는 갈래다. 둘을 가른 이유가 셋 있다.
#
# 1. **입력이 다르다.** 얼굴 상자는 `facebox.face_box()` 가 이미 잘라 준 정사각
#    크롭이라, 상자 한가운데가 정의상 얼굴이고 네 모서리는 거의 언제나 배경이다.
#    그 사실을 씨앗으로 주면 GrabCut 이 잘 맞는다.
# 2. **가중치를 안 받는다.** rembg 는 `--extra cutout` 이 있어야 하고 첫 실행에
#    네트워크를 탄다 (CA-005). 얼굴 갈래는 opencv 하나면 돌아서, 설비 없이도
#    검증과 합성을 끝까지 할 수 있다.
# 3. **앱과 모양을 맞춘다.** 앱은 ML Kit Subject Segmentation 을 쓰고 실패하면
#    타원으로 물러선다. 여기 [ellipse_alpha] 가 그 폴백과 같은 것이다 — 파이썬에서
#    본 결과가 앱에서 재현돼야 하므로 문턱·테두리 두께도 같은 값을 쓴다.
#
# 품질을 앱만큼 낼 필요는 없다. 여기서 검증할 것은 "얼굴을 어디서 어떻게 자르는가"
# 지 매팅 품질이 아니다. 더 좋은 것이 필요해지면 [grabcut] 만 갈아 끼우면 된다.
#
# ## 세 단계의 순서가 중요하다
#
#     1 배경 지우기      grabcut, 실패하면 타원
#     2 목 아래 녹이기   fade_below   ← 테두리 **전에**
#     3 테두리 띠 두르기 add_outline
#
# 앱의 `Cutout` 이 정확히 이 순서로 하고, 커밋 f229473 이 순서를 바꿨을 때 무슨
# 일이 나는지 적어 뒀다: 페이드를 먼저 하면 옅어진 알파를 따라 흰 띠가 호를 그린다.
# ---------------------------------------------------------------------------

#: "또렷한 얼굴" 로 칠 알파 문턱. 앱의 `Cutout.CORE_ALPHA` 와 같은 값이다.
#:
#: 페이드로 옅어진 가슴팍도 알파가 0 은 아니라서 [ALPHA_DUST] 로 재면 얼굴 자리에
#: 같이 들어간다. 그러면 **카드 구멍이 얼굴이 아니라 얼굴+가슴 한가운데에 맞춰져서**
#: 머리가 한쪽으로 밀린다.
CORE_ALPHA = 160

#: 테두리 띠 두께(긴 변 대비). 앱의 `Cutout.OUTLINE_RATIO` 와 같은 값.
OUTLINE_RATIO = 0.010

#: 목선 아래로 이만큼(높이 대비) 걸쳐 사라진다. 앱의 `Cutout.FADE_SPAN`.
FADE_SPAN = 0.16

#: GrabCut 반복 횟수. 5 면 충분하고 늘려도 눈에 띄게 안 좋아진다.
GRABCUT_ITERS = 5

# GrabCut 씨앗 — **정사각 크롭 안에서 머리는 타원**이라는 것을 쓴다.
#
# 처음에는 가장자리 1.5% 테두리만 배경으로 줬는데, 그러면 배경 근거가 너무 적어서
# **8장 중 7장이 "전부 앞" 으로 판정됐다.** 상자 네 모서리는 머리가 타원인 이상
# 거의 언제나 배경이라 거기를 알려 줘야 한다.
#
# 반지름은 상자 중심 기준 정규화다 (변 가운데가 1.0, 모서리가 1.414).
SEED_FG_R = 0.45   # 이 안쪽은 확실한 앞 — 얼굴 상자 한가운데는 정의상 얼굴이다
SEED_PR_FG_R = 0.95
SEED_BG_R = 1.15   # 이 바깥은 확실한 배경 — 모서리 끝자락만 걸린다

#: 오려낸 넓이가 이 범위를 벗어나면 실패로 본다.
#:
#: 얼굴 크롭이라 머리가 화면의 절반 넘게 차는 것이 정상이다. 상한을 0.95 로 뒀다가
#: 멀쩡한 결과까지 폴백으로 보냈다.
FILL_MIN, FILL_MAX = 0.25, 0.96


def grabcut(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """`(h, w, 3)` 에서 알파를 만든다. `(알파, 진짜_오려냈나)`.

    실패하면 **타원**으로 물러선다. 앱의 `Cutout.Result.Ellipse` 와 같은 폴백이고,
    이유도 같다 — 카드가 안 나오는 것이 제일 나쁘다.
    """
    cv2 = _cv2()
    h, w = rgb.shape[:2]
    r = _radius(h, w)

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[r >= SEED_BG_R] = cv2.GC_BGD
    mask[r <= SEED_PR_FG_R] = cv2.GC_PR_FGD
    mask[r <= SEED_FG_R] = cv2.GC_FGD

    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, mask, None, bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # 아무것도 못 지웠거나 다 지웠으면 실패로 본다. 둘 다 실제로 나온다 —
        # 배경이 개와 같은 색일 때 전부 앞이 되고, 역광에서 전부 뒤가 된다.
        filled = float((alpha > 0).mean())
        if not FILL_MIN <= filled <= FILL_MAX:
            raise RuntimeError(f"오려낸 넓이가 이상합니다: {filled:.2f}")
        # 가장자리 톱니를 살짝 눕힌다. 딱 자르면 오려 붙인 티가 난다.
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, min(w, h) * 0.006))
        return alpha, True
    except Exception:
        return ellipse_alpha(h, w), False


def _radius(h: int, w: int) -> np.ndarray:
    """상자 중심에서의 정규화 반지름. 변 가운데가 1.0, 모서리가 1.414."""
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    return np.sqrt(nx * nx + ny * ny)


def ellipse_alpha(h: int, w: int, feather: float = 0.06) -> np.ndarray:
    """상자를 타원으로 자른 알파. 가장자리는 흐린다.

    앱의 `Cutout.ellipseMasked()` 와 같은 폴백이다. **못 오려도 카드는 나와야 한다.**
    """
    a = np.clip((1.0 - _radius(h, w)) / max(feather, 1e-6), 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def fade_below(alpha: np.ndarray, neck: float, span: float = FADE_SPAN) -> np.ndarray:
    """[neck](높이 대비 0~1) 아래를 [span] 에 걸쳐 녹인다.

    **딱 자르지 않는 이유**: 아래가 직선으로 잘리면 흰 테두리 띠가 가로로 그어져서
    카드에 얹었을 때 "사진을 오려 붙인 것" 으로 보인다 (PR #19 이 실기기에서 본 것).
    """
    h = alpha.shape[0]
    y = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    k = np.clip(1.0 - (y - neck) / max(span, 1e-6), 0.0, 1.0)
    return (alpha.astype(np.float32) * k[:, None]).astype(np.uint8)


def add_outline(rgba: np.ndarray, ratio: float = OUTLINE_RATIO) -> np.ndarray:
    """알파를 부풀려 흰 띠를 두른다. **블러가 아니다.**

    앱이 `RenderEffect`(API 31+)를 못 쓰고 minSdk 26 이라 `room_cutout.py` 의
    `add_outline()` 을 옮긴 것과 같은 방법이다. 스티커 테두리라 카드에 어울린다.
    """
    cv2 = _cv2()
    h, w = rgba.shape[:2]
    r = max(1, int(round(max(h, w) * ratio)))
    alpha = rgba[:, :, 3]
    solid = (alpha >= ALPHA_DUST).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    grown = cv2.dilate(solid, k)

    out = np.zeros_like(rgba)
    out[:, :, :3] = 255                       # 띠는 흰색
    out[:, :, 3] = grown
    # 그 위에 원래 그림을 알파 합성으로 얹는다.
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    out[:, :, :3] = (rgba[:, :, :3] * a + out[:, :, :3] * (1 - a)).astype(np.uint8)
    out[:, :, 3] = np.maximum(grown, alpha)
    return out


def core_rect(alpha: np.ndarray, floor: int = CORE_ALPHA) -> tuple[int, int, int, int]:
    """또렷한 부분만의 사각형. 없으면 전체를 준다."""
    ys, xs = np.where(alpha >= floor)
    if len(xs) == 0:
        h, w = alpha.shape
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


@dataclass(frozen=True)
class Face:
    """카드 구멍에 끼울 얼굴 한 장.

    [core] 는 **또렷한 부분만의 사각형**이다. 앱의 `Cutout.Face` 와 같은 뜻으로,
    비트맵 전체 사각형을 구멍에 맞추면 목 아래로 흐려지는 꼬리까지 셈에 들어가서
    머리가 반대쪽으로 밀린다.
    """

    #: RGBA. 테두리를 **안 두른** 판 — 구멍에 끼울 때 쓴다
    plain: "np.ndarray"
    #: RGBA. 흰 띠를 두른 판 — 홀로 세울 때 쓴다
    outlined: "np.ndarray"
    #: `(left, top, right, bottom)` 픽셀
    core: tuple[int, int, int, int]
    #: 배경을 실제로 지웠나. False 면 타원으로 물러선 것이다
    cut: bool
    #: 구멍에 맞출 **가로 기준점**(크롭 안 픽셀). None 이면 [core] 한가운데.
    #:
    #: **알파 한가운데는 얼굴 한가운데가 아니다.** 귀가 한쪽으로 처지거나 털이
    #: 한쪽에 뭉치면 알파 상자가 그쪽으로 늘어나서, 구멍에 넣었을 때 얼굴이 반대로
    #: 밀린다 — 페키니즈에서 구멍에 뒤통수만 들어간 적이 있다. 키포인트가 있으면
    #: `facebox.face_anchor_x()` 가 코와 귀밑을 섞어 짚어 준다. **앱이 지금 못 하는
    #: 일이고, 포즈 모델을 쓰는 값어치가 여기서 나온다.**
    anchor_x: float | None = None
    #: 구멍 아래에 걸 **턱 자리**(크롭 안 픽셀). None 이면 [core] 의 아래.
    chin_y: float | None = None


def face_from(
    crop: Image.Image,
    neck: float,
    *,
    anchor_x: float | None = None,
    chin_y: float | None = None,
) -> Face:
    """잘라 낸 얼굴 한 장 → 카드에 끼울 [Face]. 위의 세 단계를 순서대로.

    @param crop `facebox.face_box()` 로 자른 정사각 크롭
    @param neck 크롭 안 0~1 세로 위치. `facebox.neck_t()` 가 준다
    @param anchor_x,chin_y 크롭 안 픽셀. 키포인트가 있으면 주는 것이 좋다
    """
    rgb = np.asarray(crop.convert("RGB"))
    alpha, ok = grabcut(rgb)
    alpha = fade_below(alpha, neck)

    plain = np.dstack([rgb, alpha])
    return Face(
        plain=plain,
        outlined=add_outline(plain),
        core=core_rect(alpha),
        cut=ok,
        anchor_x=anchor_x,
        chin_y=chin_y,
    )
