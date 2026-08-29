"""scene.json 계약 — 여기가 유일한 출처다.

필드 이름은 앱의 `ImmersiveScene` · `DexCard` 를 그대로 따라간다. 새 어휘를 만들지
않는다. 문서에 두 번 적지 않는다 — 문서가 이 파일과 어긋나면 이 파일이 맞다.

앱이 카멜케이스(`statLabel`)로 읽으므로 직렬화 이름도 카멜케이스를 쓴다. 파이썬
쪽에서는 그대로 `stat_label` 로 부르고 alias 로 오간다.
"""

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = 1

# "#RRGGBB". 앱이 Color(0xFF...) 로 파싱하므로 3자리 축약형을 허용하지 않는다.
HexColor = Annotated[str, StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$")]


class Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Dog(Base):
    """입력 3개 중 사진을 뺀 둘. 사진 자체는 JSON 에 들어가지 않는다."""

    name: str = Field(min_length=1, max_length=20)
    birthday: date


class Card(Base):
    """`DexCard` 와 이름을 맞춘 카드 한 장."""

    no: int = Field(ge=1, le=12)
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # 빈 문자열을 받는다. No.11 Tomato 는 그림에 스탯 라벨이 안 찍혀 있고, 도감은
    # "안 찍혀 있으면 지어내지 말고 비운다" 를 규칙으로 삼았다. min_length=1 로 막으면
    # 실제 카드가 계약에서 거절당한다.
    stat_label: str = Field(alias="statLabel")
    stat: int = Field(ge=0)
    foil: str = Field(min_length=1)
    art: str = Field(min_length=1)


class Rect(Base):
    """카드 크기에 대한 퍼센트. `ImmersiveScene.Fit` 과 같은 단위다."""

    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class Scene(Base):
    """`ImmersiveScene` 의 필드를 그대로 가져온 장면 값."""

    place: str = Field(min_length=1)
    seed: int
    motes: int = Field(ge=0)
    # 앞에 크게 흐리게 지나가는 잎. 앱의 ImmersiveScene 에 있는 필드다.
    leaves: int = Field(ge=0)
    dew: int = Field(ge=0)
    accent: HexColor
    accent2: HexColor

    back: Optional[str] = None
    subject: Optional[str] = None
    card: str
    frame: Optional[str] = None

    fit: Rect
    window: Rect

    # 음악은 유일한 생성·유일한 유료 단계라 실패할 수 있다. null 이면 앱의
    # SceneMusic 이 조용히 아무것도 안 한다 — 소리가 안 나도 화면은 산다.
    bgm: Optional[str] = None

    # dev 를 부르지 않기로 했으므로 산책 데이터 연동은 하지 않는다. 나중에 채울
    # 자리만 비워 둔다.
    weather: Optional[str] = None
    time_of_day: Optional[str] = None


class SceneDoc(Base):
    """scene.json 한 장 전체."""

    # pydantic v2 에서 `schema` 는 BaseModel 의 예약 이름이라 파이썬 쪽 이름을
    # 달리 두고 alias 로만 `schema` 를 쓴다.
    schema_version: int = Field(default=SCHEMA_VERSION, alias="schema")
    agent_version: str
    dog: Dog
    card: Card
    scene: Scene

    def to_json(self, *, indent: int = 2) -> str:
        """앱이 읽는 모양 그대로 직렬화한다 (alias 기준)."""
        return json.dumps(
            self.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=indent,
        ) + "\n"

    def write(self, path: Path) -> Path:
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SceneDoc":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
