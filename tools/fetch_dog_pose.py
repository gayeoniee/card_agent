"""dog-pose 데이터셋을 내려받아 `.data/dog-pose/` 에 푼다.

    uv run tools/fetch_dog_pose.py

337MB 다. 한 번만 받으면 되고 `.gitignore` 에 있어서 저장소에는 안 올라간다.

**Ultralytics 를 안 깐다.** `ultralytics` 를 부르면 데이터셋을 알아서 받아 주지만
그것 하나 때문에 AGPL-3.0 짐(그리고 torch)을 검증 단계까지 끌고 올 이유가 없다.
받는 주소는 yaml 의 `download:` 에 적힌 그 주소 그대로다.

받은 것: 이미지 6,773(train) / 1,703(val), 라벨은 YOLO pose 형식 24 키포인트.
출처는 Stanford Dogs (http://vision.stanford.edu/aditya86/ImageNetDogs/) 이고
**연구용 조건이 붙는다** — README 의 라이선스 절을 볼 것.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import urllib.request
import zipfile

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip"
ROOT = pathlib.Path(__file__).resolve().parents[1] / ".data"
DEST = ROOT / "dog-pose"

EXPECTED = {"train": 6773, "val": 1703}


def main() -> int:
    if DEST.is_dir() and all(
        len(list((DEST / "labels" / s).glob("*.txt"))) == n for s, n in EXPECTED.items()
    ):
        print(f"이미 있습니다: {DEST}")
        return 0

    ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / "dog-pose.zip"

    if not zip_path.exists():
        print(f"내려받는 중 (337MB) … {URL}")
        with urllib.request.urlopen(URL) as src, zip_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    print(f"푸는 중 → {DEST}")
    staging = ROOT / "_unzip"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)

    # zip 안이 `images/` `labels/` 로 바로 시작하는지, 한 겹 더 싸여 있는지 둘 다 받는다.
    src = staging if (staging / "labels").is_dir() else next(staging.iterdir())
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.move(str(src), str(DEST))
    shutil.rmtree(staging, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    for split, n in EXPECTED.items():
        got = len(list((DEST / "labels" / split).glob("*.txt")))
        mark = "ok" if got == n else "⚠ 다름"
        print(f"  {split}: 라벨 {got}장 (기대 {n}) {mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
