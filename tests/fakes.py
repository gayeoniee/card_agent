"""테스트용 가짜 그림. 카드 원화는 git 밖이라 저장소 안에서 만들어 쓴다."""

from PIL import Image, ImageDraw

from src.compose import _px
from src.contract import Rect


def fake_frame(window: Rect, size: tuple[int, int] = (600, 840)) -> Image.Image:
    """그림창만 투명하게 뚫린 완성 카드 흉내 (cabbage-card-frame.webp 와 같은 모양)."""
    frame = Image.new("RGBA", size, (26, 48, 30, 255))
    x, y, w, h = _px(window, size)
    frame.paste((0, 0, 0, 0), (x, y, x + w, y + h))
    return frame


def fake_dog(size: tuple[int, int] = (400, 300),
             color: tuple[int, int, int] = (214, 168, 122),
             margin: int | None = None) -> Image.Image:
    """가장자리에 투명한 여백이 있는 누끼 흉내. trim 이 도는지 보려고 여백을 둔다."""
    margin = max(4, min(size) // 8) if margin is None else margin
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((margin, margin, size[0] - margin, size[1] - margin), fill=(*color, 255))
    draw.rectangle((margin + 20, margin + 20, margin + 70, margin + 70), fill=(60, 42, 30, 255))
    return img
