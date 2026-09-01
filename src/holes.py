"""카드의 **얼굴 구멍** 표. `templates/holes.toml` 을 읽고 검사만 한다.

`compose.load_windows` 와 짝이지만 **다른 표다.** 헷갈리기 쉬운 자리라 갈라 둔다.

    windows.toml   직사각 그림창.  강아지 **전체**를 contain 으로 넣는다
                   원화는 `*-card-frame.webp` (그림 영역이 통째로 비어 있다)
    holes.toml     타원 얼굴 구멍.  강아지 **얼굴만** 넣는다
                   원화는 `*-card-slots.webp` (채소는 인쇄돼 있고 얼굴 자리만 비었다)

새로 받은 카드 12장이 뒤쪽이다. 앱 PR #19 가 코틀린으로 채소를 그려 보고 "종이 오린
꼴"이라며 되돌린 뒤, **채소는 저쪽이 그리고 우리는 자리만 비운다** 로 정한 방식이다.

숫자의 출처는 `tools/measure_holes.py` 다. 여기서 손으로 고치지 않는다.
"""

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.version import REPO_ROOT

HOLES_TOML = REPO_ROOT / "templates" / "holes.toml"

#: 얼굴 구멍이 뚫린 카드 원화. `windows.toml` 쪽의 `*-card-frame.webp` 와 다르다.
SLOTS_SUFFIX = "-card-slots.webp"


class HoleTableError(ValueError):
    """구멍 표가 카드 12장을 못 덮거나 값이 카드 밖으로 나갈 때."""


@dataclass(frozen=True)
class Hole:
    """카드 크기 대비 % 로 잡은 타원. 앱 `CardSlots.kt` 의 `Hole` 과 같은 단위다."""

    cx: float
    cy: float
    rx: float
    ry: float

    def px(self, size: tuple[int, int]) -> tuple[float, float, float, float]:
        """`(cx, cy, rx, ry)` 픽셀."""
        w, h = size
        return (self.cx / 100 * w, self.cy / 100 * h,
                self.rx / 100 * w, self.ry / 100 * h)


@dataclass(frozen=True)
class HoleCard:
    id: str
    #: 잰 원본의 픽셀 크기. 다른 해상도 판을 쓰면 여기부터 다시 재야 한다
    size: tuple[int, int]
    face: Hole
    #: 왼쪽 위 작은 원. 없는 카드도 있을 수 있다
    avatar: Hole | None

    @property
    def ratio(self) -> float:
        return self.size[0] / self.size[1]

    def art_path(self, cards_dir: Path) -> Path:
        return Path(cards_dir) / f"{self.id}{SLOTS_SUFFIX}"


def _hole(card_id: str, name: str, row: dict | None) -> Hole | None:
    if row is None:
        return None
    try:
        hole = Hole(float(row["cx"]), float(row["cy"]), float(row["rx"]), float(row["ry"]))
    except KeyError as exc:
        raise HoleTableError(f"[{card_id}] 의 {name} 에 {exc} 가 없다") from exc
    if hole.rx <= 0 or hole.ry <= 0:
        raise HoleTableError(f"[{card_id}] 의 {name} 반지름이 0 이하다")
    # 구멍이 카드 밖으로 나가면 합성이 조용히 빈 카드를 낸다. 여기서 죽는 편이 낫다.
    if not (0 <= hole.cx - hole.rx and hole.cx + hole.rx <= 100.0001
            and 0 <= hole.cy - hole.ry and hole.cy + hole.ry <= 100.0001):
        raise HoleTableError(f"[{card_id}] 의 {name} 이 카드 밖으로 나간다")
    return hole


@lru_cache(maxsize=None)
def load_holes(path: Path = HOLES_TOML) -> dict[str, HoleCard]:
    """구멍 표를 읽는다. 표가 깨져 있으면 여기서 죽는다 (뒤에서 죽는 것보다 낫다)."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    table: dict[str, HoleCard] = {}
    for card_id, row in data.items():
        try:
            size = tuple(int(v) for v in row["size"])
        except KeyError as exc:
            raise HoleTableError(f"{path}: [{card_id}] 에 {exc} 가 없다") from exc
        if len(size) != 2 or min(size) <= 0:
            raise HoleTableError(f"{path}: [{card_id}] 의 size 가 이상하다: {size}")
        face = _hole(card_id, "face", row.get("face"))
        if face is None:
            raise HoleTableError(f"{path}: [{card_id}] 에 face 가 없다")
        table[card_id] = HoleCard(
            id=card_id, size=size, face=face,
            avatar=_hole(card_id, "avatar", row.get("avatar")),
        )

    from src.crops import load_table  # 순환 import 를 피해 여기서 부른다

    missing = sorted({c.id for c in load_table().values()} - set(table))
    if missing:
        raise HoleTableError(f"{path}: {missing} 카드의 구멍이 없다")
    return table


def hole_card(card_id: str, path: Path = HOLES_TOML) -> HoleCard:
    try:
        return load_holes(path)[card_id]
    except KeyError:
        raise HoleTableError(f"{path}: 모르는 카드다: {card_id}") from None
