"""카드 12장 표 — **뽑기**다. 룰 테이블 하나로 끝나고 모델을 쓰지 않는다.

부를 때마다 12장 중 하나를 균등하게 뽑는다. 같은 강아지로 다시 부르면 **다른 카드가
나온다** — 그게 뽑기다 (CA-017). 표는 `templates/crops.toml` 이 가지고 있고 여기서는
읽고 검사만 한다.

카드 값(`no` · `name` · `statLabel` · `stat` · `foil`)의 출처는 앱의 `DexCards.kt`
다 (CA-012). 여기서 새 어휘를 만들지 않는다.
"""

import random
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.contract import Card
from src.version import REPO_ROOT

CROPS_TOML = REPO_ROOT / "templates" / "crops.toml"

#: 카드 장수. 도감 번호가 1~12 를 한 번씩 덮어야 한다.
CARD_COUNT = 12

#: 뽑기용 난수. **씨를 안 준다** — 부를 때마다 다른 값이어야 한다 (CA-017).
#: 테스트는 `draw_crop(rng=...)` 로 자기 것을 넣는다.
_RNG = random.Random()


class CropTableError(ValueError):
    """표가 카드 12장을 겹치지 않게 덮지 못할 때."""


@dataclass(frozen=True)
class Crop:
    no: int
    id: str
    korean: str
    name: str
    stat_label: str
    stat: int
    foil: str

    def to_card(self, art: str) -> Card:
        """카드 그림 파일 이름만 붙여서 계약의 Card 로 만든다."""
        return Card(
            no=self.no,
            id=self.id,
            name=self.name,
            stat_label=self.stat_label,
            stat=self.stat,
            foil=self.foil,
            art=art,
        )


@lru_cache(maxsize=None)
def load_table(path: Path = CROPS_TOML) -> dict[str, Crop]:
    """id → Crop 표를 읽는다. 표가 깨져 있으면 여기서 죽는다 (뒤에서 죽는 것보다 낫다).

    **id 로 키를 잡는다.** 카드 원화 파일 이름이 그 id 이고(`cabbage-card-slots.webp`),
    `windows.toml` · `holes.toml` 도 같은 키를 쓴다. 표마다 키가 다르면 짝을 맞추는
    코드가 표 개수만큼 늘어난다.
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("crop", [])

    table: dict[str, Crop] = {}
    for row in rows:
        try:
            crop = Crop(
                no=int(row["no"]),
                id=str(row["id"]),
                korean=str(row["korean"]),
                name=str(row["name"]),
                stat_label=str(row["statLabel"]),
                stat=int(row["stat"]),
                foil=str(row["foil"]),
            )
        except KeyError as exc:
            raise CropTableError(f"{path}: 항목에 {exc} 가 없다") from exc
        if crop.id in table:
            raise CropTableError(f"{path}: card.id 가 겹친다: {crop.id}")
        table[crop.id] = crop

    if len(table) != CARD_COUNT:
        raise CropTableError(f"{path}: 카드가 {CARD_COUNT}장이 아니다 ({len(table)}장)")

    nos = sorted(c.no for c in table.values())
    if nos != list(range(1, CARD_COUNT + 1)):
        raise CropTableError(f"{path}: 도감 번호가 1~{CARD_COUNT} 를 한 번씩 덮지 않는다 ({nos})")

    return table


def crop_by_id(card_id: str, path: Path = CROPS_TOML) -> Crop:
    """id 로 카드 하나. 뽑지 않고 **직접 고를 때** 쓴다 (`pipeline --card`)."""
    try:
        return load_table(path)[card_id]
    except KeyError:
        raise CropTableError(f"{path}: 모르는 카드 id 다: {card_id}") from None


def draw_crop(rng: random.Random | None = None, path: Path = CROPS_TOML) -> Crop:
    """12장 중 하나를 **균등하게** 뽑는다.

    @param rng 테스트가 고정할 때만 준다. 안 주면 부를 때마다 다른 값이 나온다.

    확률을 카드마다 다르게 주고 싶어지면(예: `foil` 등급) 여기 한 줄이다. 지금은
    "그냥 랜덤" 이 요청이라 균등이다.
    """
    table = load_table(path)
    # 표는 dict 라 순서가 삽입 순서다. **정렬해서 뽑는다** — 표의 줄 순서를 바꿨을 뿐인데
    # 같은 rng 가 다른 카드를 주면 테스트가 이유 없이 깨진다.
    return (rng or _RNG).choice(sorted(table.values(), key=lambda c: c.no))
