"""누끼 — 사진에서 강아지만 남긴다.

rembg 는 무겁고(onnxruntime + 가중치) 첫 실행에 모델을 받아야 해서 선택 의존성으로
갈라 두었다 (`uv sync --extra cutout`). `skin-screening` 의 `--extra model` 과 같은
방식이다. 없으면 **곱게 실패한다** — 503 성격이지 500 이 아니다.

알파가 이미 있는 PNG 를 주면 rembg 없이도 나머지 파이프라인이 그대로 돈다.
"""

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

from src.compose import trim_alpha

# 이 아래 알파는 누끼 가장자리의 먼지로 본다. 남겨 두면 bbox 가 사진 전체가 된다.
ALPHA_DUST = 8


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
