"""화풍 맞춤 — 사실 사진을 픽셀아트 쪽으로 당긴다.

`DAENGS_APP/tools/isoasset.py pastel` 이 톤을 강제로 통일하던 것과 같은 생각이다.
사진과 픽셀아트 프레임이 한 화면에서 붙는지는 **사람이 봐야** 판정되므로(앱 협업규칙
1절), 여기서는 고르라고 옵션을 여러 개 만들어 둔다. 고르는 일은
`tools/contact_sheet.py` 가 뽑아 주는 시트를 보고 사람이 한다.
"""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class PixelStyle:
    """화풍 손잡이 한 벌. 이름은 비교 시트의 칸 제목이 된다."""

    name: str
    # 몇 픽셀을 한 칸으로 뭉칠지. 1 이면 뭉치지 않는다.
    block: int = 1
    # 양자화 색 수. 0 이면 하지 않는다.
    colors: int = 0
    # 포스터라이즈 비트 (1~8). 0 이면 하지 않는다.
    posterize: int = 0
    saturation: float = 1.0
    contrast: float = 1.0
    # 파스텔 쪽으로 끌어당기는 정도 (0~1). isoasset.py pastel 과 같은 취지.
    pastel: float = 0.0
    # 픽셀아트에는 반투명 가장자리가 없다. 알파를 이 문턱으로 자른다.
    alpha_cut: int = 128
    dither: bool = False


# 비교 시트에 나란히 세울 후보들. 사람이 실기기에서 보고 고른다.
STYLES: dict[str, PixelStyle] = {
    "raw": PixelStyle("raw", alpha_cut=0),
    "soft": PixelStyle("soft", colors=48, saturation=1.05, pastel=0.12),
    "pixel": PixelStyle("pixel", block=3, colors=32, saturation=1.1, contrast=1.05),
    "pixel-hard": PixelStyle("pixel-hard", block=5, colors=16, saturation=1.2, contrast=1.15),
    "poster": PixelStyle("poster", block=2, posterize=3, saturation=1.15),
    "pastel": PixelStyle("pastel", block=3, colors=24, pastel=0.28, saturation=0.95),
}

DEFAULT_STYLE = "pixel"


def style_by_name(name: str) -> PixelStyle:
    try:
        return STYLES[name]
    except KeyError:
        raise ValueError(f"모르는 화풍이다: {name} (있는 것: {', '.join(STYLES)})") from None


def _pastelize(rgb: Image.Image, amount: float) -> Image.Image:
    """흰 쪽으로 섞어 톤을 눕힌다. 원화의 파스텔 톤에 맞추려는 것."""
    arr = np.asarray(rgb, dtype=np.float32)
    arr = arr * (1.0 - amount) + 255.0 * amount
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _blockify(img: Image.Image, block: int, *, alpha: bool) -> Image.Image:
    """줄였다 최근접으로 늘려 칸을 만든다. 크기는 그대로 돌아온다."""
    w, h = img.size
    small = (max(1, w // block), max(1, h // block))
    # 색은 평균으로 줄여야 덩어리가 곱게 지고, 알파는 최근접이라야 가장자리가 안 번진다.
    down = img.resize(small, Image.Resampling.NEAREST if alpha else Image.Resampling.BOX)
    return down.resize((w, h), Image.Resampling.NEAREST)


def apply_style(img: Image.Image, style: PixelStyle) -> Image.Image:
    """RGBA 한 장에 화풍을 입힌다. 크기는 바뀌지 않는다."""
    rgba = img.convert("RGBA")
    rgb = rgba.convert("RGB")
    alpha = rgba.getchannel("A")

    if style.alpha_cut > 0:
        alpha = alpha.point(lambda v: 255 if v >= style.alpha_cut else 0)

    if style.saturation != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(style.saturation)
    if style.contrast != 1.0:
        rgb = ImageEnhance.Contrast(rgb).enhance(style.contrast)
    if style.pastel > 0:
        rgb = _pastelize(rgb, style.pastel)

    if style.block > 1:
        rgb = _blockify(rgb, style.block, alpha=False)
        alpha = _blockify(alpha, style.block, alpha=True)

    if style.posterize:
        rgb = ImageOps.posterize(rgb, max(1, min(8, style.posterize)))

    if style.colors:
        dither = Image.Dither.FLOYDSTEINBERG if style.dither else Image.Dither.NONE
        # 투명한 자리의 색이 팔레트를 갉아먹지 않도록 알파 밖은 보지 않게 하고 싶지만,
        # Pillow 의 quantize 는 마스크를 받지 않는다. 대신 투명한 자리를 이미 보이는
        # 색의 평균으로 덮어 두어 팔레트가 엉뚱한 색을 잡지 않게 한다.
        rgb = _fill_transparent_with_mean(rgb, alpha)
        rgb = rgb.quantize(colors=max(2, style.colors), method=Image.Quantize.MEDIANCUT,
                           dither=dither).convert("RGB")

    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def _fill_transparent_with_mean(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    mask = np.asarray(alpha, dtype=np.uint8) > 0
    if mask.all() or not mask.any():
        return rgb
    arr = np.asarray(rgb, dtype=np.uint8).copy()
    arr[~mask] = arr[mask].mean(axis=0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def variants(names: list[str] | None = None) -> list[PixelStyle]:
    """비교 시트에 세울 후보 목록."""
    if names is None:
        return list(STYLES.values())
    return [style_by_name(n) for n in names]


def scaled(style: PixelStyle, factor: float) -> PixelStyle:
    """카드 안 그림창 크기에 맞춰 칸 크기를 비례로 키운다.

    블록은 픽셀 수라서, 그림창이 큰 카드에서 같은 값을 쓰면 칸이 상대적으로 작아진다.
    화면에서 보이는 칸 크기를 맞추려면 그림창 크기에 비례해야 한다.
    """
    return replace(style, block=max(1, round(style.block * factor)))
