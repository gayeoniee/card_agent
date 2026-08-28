"""음악 provider 인터페이스 — 함수 하나짜리다.

일부러 좁게 잡았다. 실제 서비스가 정해지지 않았으므로 (상업 이용 라이선스 확인이
그 카드의 첫 항목이다) 넓은 인터페이스를 먼저 만들면 대개 틀린다.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


class MusicUnavailable(RuntimeError):
    """서비스가 없거나·막혔거나·돈이 없다. 카드는 그대로 나가야 한다."""


@runtime_checkable
class MusicProvider(Protocol):
    def generate(self, prompt: str, seconds: int) -> bytes:
        """음원 바이트를 돌려준다.

        포맷은 soundfile 이 읽을 수 있는 것이면 된다(WAV 를 기대한다). 루프 이음새와
        OGG 변환은 provider 가 아니라 `src/loop.py` 가 맡는다 — 서비스가 바뀌어도
        루프 규격은 그대로여야 하기 때문이다.

        실패는 MusicUnavailable 로 올린다.
        """


def prompt_for(crop_korean: str, dog_name: str) -> str:
    """장면에 맞는 프롬프트 한 줄. 서비스가 정해지면 여기부터 손본다."""
    return (
        f"조용한 새벽 텃밭, {crop_korean} 이랑 사이. 작은 강아지 {dog_name} 이(가) 걷는 장면. "
        "느린 템포의 부드러운 앰비언트, 8비트 음색 약간, 드럼 없음, 반복되는 짧은 루프."
    )


def save_bytes(data: bytes, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
