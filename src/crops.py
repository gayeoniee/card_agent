"""생일 월 → 작물 카드. 룰 테이블 하나로 끝난다 — 모델을 쓰지 않는다.

카드가 정확히 12장이고 달도 12개라 1:1 로 떨어진다. 표는 templates/crops.toml 이
가지고 있고 여기서는 읽고 검사만 한다.
"""

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.contract import Card
from src.version import REPO_ROOT

CROPS_TOML = REPO_ROOT / "templates" / "crops.toml"


class CropTableError(ValueError):
    """표가 12달을 빠짐없이·겹치지 않게 덮지 못할 때."""


@dataclass(frozen=True)
class Crop:
    month: int
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
def load_table(path: Path = CROPS_TOML) -> dict[int, Crop]:
    """월 → Crop 표를 읽는다. 표가 깨져 있으면 여기서 죽는다 (뒤에서 죽는 것보다 낫다)."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("crop", [])

    table: dict[int, Crop] = {}
    for row in rows:
        try:
            crop = Crop(
                month=int(row["month"]),
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
        if crop.month in table:
            raise CropTableError(f"{path}: {crop.month}월이 두 번 나온다")
        table[crop.month] = crop

    missing = sorted(set(range(1, 13)) - set(table))
    if missing:
        raise CropTableError(f"{path}: {missing} 월이 비어 있다")

    nos = [c.no for c in table.values()]
    if len(set(nos)) != 12 or sorted(nos) != list(range(1, 13)):
        raise CropTableError(f"{path}: 도감 번호가 1~12 를 한 번씩 덮지 않는다 ({sorted(nos)})")

    ids = [c.id for c in table.values()]
    if len(set(ids)) != 12:
        raise CropTableError(f"{path}: card.id 가 겹친다")

    return table


def crop_for_month(month: int, path: Path = CROPS_TOML) -> Crop:
    if not 1 <= month <= 12:
        raise ValueError(f"월은 1~12 여야 한다: {month}")
    return load_table(path)[month]


def crop_for_birthday(birthday, path: Path = CROPS_TOML) -> Crop:
    """생일(date) 하나면 카드가 정해진다. 날짜와 연도는 보지 않는다."""
    return crop_for_month(birthday.month, path)
