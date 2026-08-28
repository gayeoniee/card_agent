"""테스트에서 `from src...` 가 되도록 저장소 뿌리를 경로에 넣는다.

card_agent 는 설치되는 패키지가 아니라 폴더째 옮기는 물건이라
(pyproject 의 `[tool.uv] package = false`) 설치 대신 경로로 푼다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
