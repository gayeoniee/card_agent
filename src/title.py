"""제목바의 글자를 지우고 다시 찍는다.

카드의 이름·번호는 **그림에 인쇄돼 있다.** `cards.mjs` 의 텍스트는 그 사본일 뿐이라
(`"지어낸 게 아니라 그림 안에 이미 인쇄돼 있는 문구를 그대로 옮긴 것"`), 강아지 이름을
카드에 찍으려면 그림을 고치는 수밖에 없다.

제목바는 **어두운 세로 그라디언트 위의 흰 세리프**라 지울 수 있는 조건이다. 흰 글자를
잡아 넉넉히 부풀리면 검은 테두리까지 덮이고, 남은 구멍을 biharmonic 으로 메우면
바탕이 이어진다.

⚠ **이건 그림을 고치는 일이라 사람이 봐야 한다** (앱 협업규칙 1절). 글자 얼굴·크기·
자간이 원본과 붙는지, 지운 자리가 티 나는지는 만들어서 카드 크기로 본다.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.contract import Rect
from src.version import REPO_ROOT

FONTS_DIR = REPO_ROOT / "templates" / "fonts"

# 이 위로 밝으면 글자로 본다. 바가 어둡고 글자가 흰색이라 넉넉히 갈린다.
GLYPH_LUMA = 120
# 글자에서 몇 px 까지 부풀릴지 (홀수). 검은 테두리가 이 안에 들어온다.
GROW = 13


class TitleUnavailable(RuntimeError):
    """인페인팅 설비(scikit-image)가 없다. 카드는 원래 제목 그대로 나간다."""


@dataclass(frozen=True)
class TitleStyle:
    font: str = "Cinzel-Bold.ttf"
    korean_font: str = "NotoSansKR-Bold.ttf"
    # 바 높이에 대한 글자 크기 비율. 원본에 맞춰 사람이 고른 값이다.
    size_ratio: float = 0.76
    korean_size_ratio: float = 0.62
    dy_ratio: float = -0.03
    fill: tuple[int, int, int] = (252, 252, 250)
    stroke: tuple[int, int, int] = (12, 16, 10)

    def font_for(self, text: str) -> str:
        return self.korean_font if any("가" <= ch <= "힣" for ch in text) else self.font

    def ratio_for(self, text: str) -> float:
        return self.korean_size_ratio if any("가" <= ch <= "힣" for ch in text) else self.size_ratio


def _px(rect: Rect, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    return (round(rect.x / 100 * w), round(rect.y / 100 * h),
            round((rect.x + rect.w) / 100 * w), round((rect.y + rect.h) / 100 * h))


def blank_title(img: Image.Image, box: Rect) -> Image.Image:
    """제목바에서 글자만 지우고 바탕을 잇는다."""
    try:
        from skimage.restoration import inpaint_biharmonic
    except ImportError as exc:
        raise TitleUnavailable(
            "제목을 지우려면 scikit-image 가 필요하다: uv sync --extra title"
        ) from exc

    # 알파를 따로 떼어 둔다. RGB 로만 작업하고 끝에 도로 붙이지 않으면 프레임의
    # 그림창(투명한 구멍)이 불투명해져서 강아지가 통째로 가린다.
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")
    arr = np.asarray(rgb).astype(np.float64)
    x0, y0, x1, y1 = _px(box, rgb.size)
    sub = arr[y0:y1, x0:x1]
    if sub.size == 0:
        raise TitleUnavailable(f"제목바가 비었다: {box}")

    glyph = sub.mean(axis=2) > GLYPH_LUMA
    mask = np.asarray(
        Image.fromarray((glyph * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(GROW))
    ) > 0
    if not mask.any():
        return img.copy()

    arr[y0:y1, x0:x1] = inpaint_biharmonic(sub / 255.0, mask, channel_axis=-1) * 255
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    out.putalpha(alpha)
    return out


def draw_title(img: Image.Image, box: Rect, text: str,
               style: TitleStyle = TitleStyle(), fonts_dir: Path = FONTS_DIR) -> Image.Image:
    """빈 제목바에 이름을 찍는다. 바 가운데 정렬."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = _px(box, out.size)
    bar_h = y1 - y0

    path = Path(fonts_dir) / style.font_for(text)
    if not path.exists():
        raise TitleUnavailable(f"폰트가 없다: {path} (README 의 '카드에 이름 찍기' 참고)")
    size = max(8, round(bar_h * style.ratio_for(text)))
    font = ImageFont.truetype(str(path), size)

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    # 글자가 바보다 넓으면 줄인다 — 긴 이름이 은색 테두리를 넘으면 안 된다.
    while right - left > (x1 - x0) * 0.94 and size > 8:
        size -= 2
        font = ImageFont.truetype(str(path), size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)

    x = (x0 + x1) // 2 - (right + left) // 2
    y = (y0 + y1) // 2 - (bottom + top) // 2 + round(bar_h * style.dy_ratio)
    draw.text((x, y), text, font=font, fill=style.fill,
              stroke_width=max(2, size // 14), stroke_fill=style.stroke)
    return out


def personalize_title(img: Image.Image, box: Rect, text: str,
                      style: TitleStyle = TitleStyle(), fonts_dir: Path = FONTS_DIR) -> Image.Image:
    return draw_title(blank_title(img, box), box, text, style, fonts_dir)
