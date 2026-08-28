"""카드 합성 — 사진을 카드 그림창에 contain 으로 맞춰 얹는다.

방향을 뒤집은 곳이다. 그림을 놓고 사람이 `fit` 을 재는 대신, 카드별 그림창을
`templates/windows.toml` 에 상수로 못박고 누끼 bbox 를 그 창에 맞춘다. 그러면 `fit`
은 사람이 재는 값이 아니라 **코드가 뱉는 값**이 된다.

카드 원화는 그림 영역만 투명하게 지워 둔 완성 카드(`cabbage-card-frame.webp` 와 같은
모양)라, 강아지를 먼저 깔고 그 위에 원화를 얹으면 그림창 안에만 보인다.
"""

import math
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from src.contract import Rect
from src.pixelize import PixelStyle, apply_style, scaled
from src.version import REPO_ROOT

WINDOWS_TOML = REPO_ROOT / "templates" / "windows.toml"
CARDS_DIR = REPO_ROOT / "templates" / "cards"

# 그림창 좌표를 잰 기준 카드의 짧은 변(px). 화풍의 칸 크기를 카드 크기에 비례로
# 맞출 때 쓴다 — 큰 카드에서 같은 block 을 쓰면 화면에서 칸이 작아 보인다.
STYLE_REFERENCE_EDGE = 512


class WindowTableError(ValueError):
    """그림창 표가 카드 12장을 못 덮을 때."""


class CardArtMissing(FileNotFoundError):
    """카드 원화가 git 밖(templates/cards/)이라 없을 수 있다. 곱게 알린다."""


@dataclass(frozen=True)
class CardTemplate:
    id: str
    window: Rect
    frame: str

    def frame_path(self, cards_dir: Path = CARDS_DIR) -> Path:
        return Path(cards_dir) / self.frame


@lru_cache(maxsize=None)
def load_windows(path: Path = WINDOWS_TOML) -> dict[str, CardTemplate]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, CardTemplate] = {}
    for card_id, row in data.items():
        try:
            window = Rect(x=float(row["x"]), y=float(row["y"]),
                          w=float(row["w"]), h=float(row["h"]))
        except KeyError as exc:
            raise WindowTableError(f"{path}: [{card_id}] 에 {exc} 가 없다") from exc
        if not (0 <= window.x and 0 <= window.y
                and window.x + window.w <= 100.0001
                and window.y + window.h <= 100.0001):
            raise WindowTableError(f"{path}: [{card_id}] 그림창이 카드 밖으로 나간다")
        frame = str(row.get("frame") or f"{card_id}-card-frame.webp")
        table[card_id] = CardTemplate(id=card_id, window=window, frame=frame)

    from src.crops import load_table  # 순환 import 를 피해 여기서 부른다

    missing = sorted({c.id for c in load_table().values()} - set(table))
    if missing:
        raise WindowTableError(f"{path}: {missing} 카드의 그림창이 없다")
    return table


def template_for(card_id: str, path: Path = WINDOWS_TOML) -> CardTemplate:
    try:
        return load_windows(path)[card_id]
    except KeyError:
        raise WindowTableError(f"{path}: 모르는 카드다: {card_id}") from None


def trim_alpha(img: Image.Image, *, threshold: int = 8) -> Image.Image:
    """알파 bbox 로 잘라 낸다 (neo-hologram-layers.py 의 trim 과 같은 일).

    누끼 가장자리에 알파 1~2 짜리 먼지가 남으면 bbox 가 사진 전체가 되어 버려서,
    문턱을 두고 자른다.
    """
    rgba = img.convert("RGBA")
    mask = rgba.getchannel("A").point(lambda v: 255 if v >= threshold else 0)
    box = mask.getbbox()
    return rgba if box is None else rgba.crop(box)


def _px(rect: Rect, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """퍼센트 사각형을 픽셀로. 안쪽으로 깎는다.

    반올림하면 창이 원래보다 반 픽셀 넓어질 수 있고, 그러면 그림이 원화의 투명한
    구멍 밖으로 한 줄 삐져나온다. 시작은 올리고 끝은 내려서 항상 창 안에 들어가게
    한다 — 그래야 fit ⊆ window 가 퍼센트로도 성립한다.
    """
    w, h = size
    x0 = math.ceil(rect.x / 100 * w)
    y0 = math.ceil(rect.y / 100 * h)
    x1 = math.floor((rect.x + rect.w) / 100 * w)
    y1 = math.floor((rect.y + rect.h) / 100 * h)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _pct(box: tuple[int, int, int, int], size: tuple[int, int]) -> Rect:
    w, h = size
    x, y, bw, bh = box
    return Rect(
        x=round(x / w * 100, 2),
        y=round(y / h * 100, 2),
        w=round(bw / w * 100, 2),
        h=round(bh / h * 100, 2),
    )


def contain(subject: tuple[int, int], window: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """subject 를 window 안에 비율 그대로, 가운데에 넣는다 (px)."""
    sw, sh = subject
    wx, wy, ww, wh = window
    if sw <= 0 or sh <= 0:
        raise ValueError("빈 그림은 맞출 수 없다")
    scale = min(ww / sw, wh / sh)
    nw = max(1, round(sw * scale))
    nh = max(1, round(sh * scale))
    return (wx + (ww - nw) // 2, wy + (wh - nh) // 2, nw, nh)


def round_corners(img: Image.Image, radius_pct: float) -> Image.Image:
    """카드 모서리를 알파로 깎는다 (neo-hologram-layers.py 의 card 모드)."""
    if radius_pct <= 0:
        return img
    rgba = img.convert("RGBA")
    w, h = rgba.size
    radius = round(min(w, h) * radius_pct / 100)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    out = rgba.copy()
    out.putalpha(Image.composite(rgba.getchannel("A"), Image.new("L", (w, h), 0), mask))
    return out


@dataclass(frozen=True)
class Composed:
    card: Image.Image       # 카드 한 장 (그림 + 프레임)
    subject: Image.Image    # 화풍을 입힌 강아지만 (이머시브 subject 로 쓴다)
    fit: Rect               # 코드가 뱉은 값
    window: Rect            # 상수로 못박은 그림창


def compose_card(
    frame: Image.Image,
    subject: Image.Image,
    window: Rect,
    *,
    style: PixelStyle | None = None,
    background: tuple[int, int, int, int] | None = None,
    corner_radius_pct: float = 0.0,
) -> Composed:
    """완성 카드 원화의 투명한 그림창 자리에 강아지를 깔고 원화를 얹는다.

    화풍은 **그림창 크기로 줄인 뒤에** 입힌다. 먼저 입히고 줄이면 애써 만든 칸이
    보간으로 뭉개져서, 화면에서 보이는 칸 크기가 카드마다 달라진다.
    """
    frame = frame.convert("RGBA")
    size = frame.size
    win_px = _px(window, size)
    trimmed = trim_alpha(subject)
    box = contain(trimmed.size, win_px)

    art = trimmed.resize((box[2], box[3]), Image.Resampling.LANCZOS)
    if style is not None:
        factor = min(size) / STYLE_REFERENCE_EDGE
        art = apply_style(art, scaled(style, factor))

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    if background is not None:
        ImageDraw.Draw(canvas).rectangle(
            (win_px[0], win_px[1], win_px[0] + win_px[2] - 1, win_px[1] + win_px[3] - 1),
            fill=background,
        )
    canvas.alpha_composite(art, (box[0], box[1]))
    card = Image.alpha_composite(canvas, frame)
    card = round_corners(card, corner_radius_pct)

    return Composed(card=card, subject=art, fit=_pct(box, size), window=window)


def open_frame(template: CardTemplate, cards_dir: Path = CARDS_DIR) -> Image.Image:
    path = template.frame_path(cards_dir)
    if not path.exists():
        raise CardArtMissing(
            f"카드 원화가 없다: {path}. 원화는 git 밖에 두므로 --assets 로 위치를 준다"
        )
    return Image.open(path).convert("RGBA")


def save_webp(img: Image.Image, path: Path, *, quality: int = 92) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 알파가 있는 카드라 lossless 를 쓰지 않으면 그림창 가장자리가 지저분해진다.
    img.save(path, format="WEBP", lossless=True, quality=quality, method=6)
    return path
