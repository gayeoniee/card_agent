"""카드 합성 — 사진을 카드 그림창에 contain 으로 맞춰 얹는다.

방향을 뒤집은 곳이다. 그림을 놓고 사람이 `fit` 을 재는 대신, 카드별 그림창을 상수로
못박고 누끼 bbox 를 그 창에 맞춘다. 그러면 `fit` 은 사람이 재는 값이 아니라
**코드가 뱉는 값**이 된다.

카드 원화는 그림 영역만 투명하게 지워 둔 완성 카드(`cabbage-card-frame.webp` 와 같은
모양)라, 강아지를 먼저 깔고 그 위에 원화를 얹으면 그림창 안에만 보인다.

**사각형이 세 개다. 헷갈리면 개가 잘린다.**

| 이름 | 무엇 | 누가 정하나 |
| --- | --- | --- |
| `art` | 프레임에 뚫린 **구멍**. 사진이 여기 들어간다 | 프레임 알파에서 뽑는다 (`extract_art_window`) |
| `fit` | 카드 안에서 **누끼가 차지한 자리** | 코드가 뱉는다 |
| `window` | 앱의 **창 벌어지는 연출** 좌표 | 앱이 정한 상수. 그대로 넘긴다 |

앱의 `immersive.mjs` 가 `fit` 을 "카드 안에서 누끼가 차지하는 자리"로, `window` 를
`setWindow()` 의 클립 사각형(`--win-ox/oy`)으로 쓴다. 배추에서 둘은 서로 다른 값이고
(`fit` 6.06/14.15/87.43/62.70 · `window` 4.91/10.28/90.51/81.58), 구멍은 `fit` 쪽에
가깝다. `window` 에 맞춰 넣으면 개가 세로로 커져 프레임에 잘린다.
"""

import math
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
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
    art: Rect        # 사진이 들어갈 구멍
    window: Rect     # 앱의 창 연출 좌표 (우리가 정하는 값이 아니다)
    frame: str
    measured: bool   # 진짜 프레임에서 뽑은 값인가, 아직 자리값인가
    inset: float = 0.0      # 구멍에서 안쪽으로 비울 %
    anchor_y: float = 0.5   # 누끼 가운데가 구멍 높이의 어디에 오는지
    title: Rect | None = None   # 제목바. 없으면 이름을 못 찍는다

    def frame_path(self, cards_dir: Path = CARDS_DIR) -> Path:
        return Path(cards_dir) / self.frame


def _rect(card_id: str, key: str, row: dict, path: Path) -> Rect:
    try:
        raw = row[key]
        rect = Rect(x=float(raw["x"]), y=float(raw["y"]),
                    w=float(raw["w"]), h=float(raw["h"]))
    except KeyError as exc:
        raise WindowTableError(f"{path}: [{card_id}] 의 {key} 에 {exc} 가 없다") from exc
    if not (0 <= rect.x and 0 <= rect.y
            and rect.x + rect.w <= 100.0001
            and rect.y + rect.h <= 100.0001):
        raise WindowTableError(f"{path}: [{card_id}] 의 {key} 가 카드 밖으로 나간다")
    return rect


@lru_cache(maxsize=None)
def load_windows(path: Path = WINDOWS_TOML) -> dict[str, CardTemplate]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, CardTemplate] = {}
    for card_id, row in data.items():
        frame = str(row.get("frame") or f"{card_id}-card-frame.webp")
        table[card_id] = CardTemplate(
            id=card_id,
            art=_rect(card_id, "art", row, path),
            window=_rect(card_id, "window", row, path),
            frame=frame,
            measured=bool(row.get("measured", False)),
            inset=float(row.get("inset", 0.0)),
            anchor_y=float(row.get("anchor_y", 0.5)),
            title=_rect(card_id, "title", row, path) if "title" in row else None,
        )

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


def contain(subject: tuple[int, int], window: tuple[int, int, int, int],
            *, inset: float = 0.0, anchor_y: float = 0.5) -> tuple[int, int, int, int]:
    """subject 를 window 안에 비율 그대로 넣는다 (px).

    구멍에 꽉 채워 정중앙에 놓는 게 기본이지만, 앱의 배추는 그렇게 놓여 있지 않다.
    구멍보다 3.3% 작고 세로로는 조금 위에 앉는다 (`immersive.css` 의
    `transform-origin: … 46%` 와 같은 자리). 그래서 손잡이를 둘 둔다.

    - `inset`  : 구멍에서 안쪽으로 몇 % 를 비울지. 얼굴이 테두리에 닿지 않게 한다
    - `anchor_y`: 누끼의 **가운데**가 구멍 높이의 어디에 오는지 (0~1). 0.5 면 정중앙

    둘 다 그림 판단이라 카드마다 사람이 볼 값이다 (앱 협업규칙 1절). 기본값은
    "꽉 채워 가운데" 로 두고, 아는 카드만 표에 적는다.
    """
    sw, sh = subject
    wx, wy, ww, wh = window
    if sw <= 0 or sh <= 0:
        raise ValueError("빈 그림은 맞출 수 없다")

    iw = ww * (1 - inset / 100)
    ih = wh * (1 - inset / 100)
    scale = min(iw / sw, ih / sh)
    nw = max(1, round(sw * scale))
    nh = max(1, round(sh * scale))

    x = wx + (ww - nw) // 2
    y = round(wy + wh * anchor_y - nh / 2)
    # 구멍 밖으로는 안 나간다 — 나가면 프레임에 잘린다.
    y = max(wy, min(y, wy + wh - nh))
    return (x, y, nw, nh)


def _holes(frame: Image.Image, threshold: int) -> Image.Image:
    """바깥에서 이어진 투명(모서리 라운딩)을 지우고 남은 안쪽 구멍만 남긴다."""
    rgba = frame.convert("RGBA")
    w, h = rgba.size
    mask = rgba.getchannel("A").point(lambda v: 255 if v < threshold else 0)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if mask.getpixel(seed):
            ImageDraw.floodfill(mask, seed, 0)
    return mask


def extract_backing_box(frame: Image.Image, *, threshold: int = 200) -> Rect:
    """뒤가 비치는 자리 **전부**를 감싸는 상자. 앱이 `window` 라고 부르는 값이다.

    모서리만 빼고 다 훑는다. 창이 뚫린 데보다 커도 상관없다 — 넘치는 만큼은 틀의
    불투명한 부분이 가린다.
    """
    mask = _holes(frame, threshold)
    box = mask.getbbox()
    if box is None:
        raise WindowTableError("프레임에 뚫린 곳이 없다 — 그림창이 지워진 원화가 맞나")
    left, top, right, bottom = box
    return _pct((left, top, right - left, bottom - top), frame.size)


def _largest_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    """True 로만 채워진 제일 넓은 직사각형 (x, y, w, h).

    행마다 "위로 몇 칸 연속인가" 히스토그램을 만들고, 그 히스토그램의 최대 직사각형을
    스택으로 찾는 고전적인 방법이다.
    """
    h, w = mask.shape
    heights = np.zeros(w + 1, dtype=np.int64)   # 끝에 0 하나를 두면 스택이 저절로 비워진다
    best = (0, (0, 0, 0, 0))

    for y in range(h):
        row = mask[y]
        heights[:w] = np.where(row, heights[:w] + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            while stack and heights[stack[-1]] >= heights[x]:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                height = int(heights[top])
                width = x - left
                area = height * width
                if area > best[0]:
                    best = (area, (left, y - height + 1, width, height))
            stack.append(x)

    if best[0] == 0:
        raise WindowTableError("프레임에 뚫린 구멍이 없다 — 그림창이 지워진 원화가 맞나")
    return best[1]


def _grow(mask: np.ndarray, box: tuple[int, int, int, int],
          min_fraction: float) -> tuple[int, int, int, int]:
    """꽉 찬 직사각형에서 시작해 "거의 다 투명한" 줄까지 넓힌다.

    그림창은 모서리가 둥글어서 위아래 몇 줄이 100% 투명하지 않다. 꽉 찬 직사각형만
    쓰면 그만큼 짧게 잡힌다. 반대로 아무 줄이나 먹으면 바로 아래 붙어 있는 기술 줄
    (`LEAFY LOOK`)까지 삼킨다 — 배추에서 재보면 그림창 본체는 가로로 89% 투명이고
    기술 줄은 27% 라 그 사이에서 갈린다.
    """
    h, w = mask.shape
    x, y, bw, bh = box
    for _ in range(2):
        while y > 0 and mask[y - 1, x:x + bw].mean() >= min_fraction:
            y -= 1
            bh += 1
        while y + bh < h and mask[y + bh, x:x + bw].mean() >= min_fraction:
            bh += 1
        while x > 0 and mask[y:y + bh, x - 1].mean() >= min_fraction:
            x -= 1
            bw += 1
        while x + bw < w and mask[y:y + bh, x + bw].mean() >= min_fraction:
            bw += 1
    return (x, y, bw, bh)


def extract_art_window(frame: Image.Image, *, threshold: int = 200,
                       min_fraction: float = 0.6) -> Rect:
    """사진이 들어갈 **그림창** 하나만 뽑는다. 사람이 자를 댈 필요가 없다.

    구멍 전부의 경계상자를 잡으면 안 된다. 틀에는 그림창 말고도 뚫린 데가 있다 —
    원래 카드에서 그림 위에 얹혀 있던 반투명 판(제목 오른쪽 판 · 기술 줄 · 스탯 줄)
    자리라, 그림을 지울 때 뒤가 같이 비었기 때문이다(`cards.mjs` 에 배추의 실측이
    적혀 있다). 그걸 다 삼키면 상자가 카드 전체만 해지고, 사진이 거기 맞춰 커져
    얼굴이 인쇄물 뒤로 가린다.

    그 판들은 그림창과 **이어져 있어서** 성분으로도 안 갈린다 — 한 곳에서 물을 부으면
    구멍이 전부 사라진다. 그래서 **꽉 찬 제일 넓은 직사각형**을 찾는다. 판들은 얇아서
    면적으로 밀리고, 세로 한 줄만 훑는 방법과 달리 판이 그림창보다 길어도 안 속는다.
    """
    mask = np.asarray(_holes(frame, threshold), dtype=np.uint8) > 0
    return _pct(_grow(mask, _largest_rect(mask), min_fraction), frame.size)


def _cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """비율을 지키며 상자를 다 덮게 키우고 넘치는 만큼 가운데에서 잘라 낸다."""
    tw, th = size
    scale = max(tw / img.width, th / img.height)
    big = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                     Image.Resampling.LANCZOS)
    left = (big.width - tw) // 2
    top = (big.height - th) // 2
    return big.crop((left, top, left + tw, top + th))


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
    fit: Rect               # 코드가 뱉은 값 — 카드 안에서 누끼가 차지한 자리
    art_window: Rect        # 사진을 넣은 구멍 (앱의 window 와는 다른 값이다)


def compose_card(
    frame: Image.Image,
    subject: Image.Image,
    art_window: Rect,
    *,
    style: PixelStyle | None = None,
    background: tuple[int, int, int, int] | None = None,
    back: Image.Image | None = None,
    corner_radius_pct: float = 0.0,
    inset: float = 0.0,
    anchor_y: float = 0.5,
) -> Composed:
    """완성 카드 원화의 투명한 그림창 자리에 강아지를 깔고 원화를 얹는다.

    화풍은 **그림창 크기로 줄인 뒤에** 입힌다. 먼저 입히고 줄이면 애써 만든 칸이
    보간으로 뭉개져서, 화면에서 보이는 칸 크기가 카드마다 달라진다.
    """
    frame = frame.convert("RGBA")
    size = frame.size
    win_px = _px(art_window, size)
    trimmed = trim_alpha(subject)
    box = contain(trimmed.size, win_px, inset=inset, anchor_y=anchor_y)

    art = trimmed.resize((box[2], box[3]), Image.Resampling.LANCZOS)
    if style is not None:
        factor = min(size) / STYLE_REFERENCE_EDGE
        art = apply_style(art, scaled(style, factor))

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    if back is not None:
        # 프레임은 인쇄 배경까지 지워져 있어서 구멍이 그냥 비어 있다. 뒷그림을 구멍에
        # **덮도록**(cover) 깔아야 원본 카드처럼 보인다 — contain 으로 깔면 가장자리가
        # 빈 채로 남는다.
        canvas.alpha_composite(_cover(back.convert("RGBA"), win_px[2:]), win_px[:2])
    if background is not None:
        ImageDraw.Draw(canvas).rectangle(
            (win_px[0], win_px[1], win_px[0] + win_px[2] - 1, win_px[1] + win_px[3] - 1),
            fill=background,
        )
    canvas.alpha_composite(art, (box[0], box[1]))
    card = Image.alpha_composite(canvas, frame)
    card = round_corners(card, corner_radius_pct)

    return Composed(card=card, subject=art, fit=_pct(box, size), art_window=art_window)


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
