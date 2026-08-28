"""agent_version 의 출처.

pyproject.toml 의 version 한 줄이 dev 로 폴더째 옮겼을 때의 추적 수단이라
버전을 코드에 두 번 적지 않는다. 여기서 pyproject 를 읽어서 쓴다.
"""

import tomllib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# pyproject 를 못 찾는 배포 형태(폴더만 복사 등)에서 쓰는 값. 읽기가 성공하면
# 이 값은 쓰이지 않는다.
FALLBACK_VERSION = "0.0.0+unknown"


@lru_cache(maxsize=1)
def agent_version() -> str:
    """pyproject.toml 의 [project].version 을 읽어서 돌려준다."""
    path = REPO_ROOT / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except (OSError, tomllib.TOMLDecodeError, KeyError):
        return FALLBACK_VERSION
