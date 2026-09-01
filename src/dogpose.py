"""dog-pose 데이터셋 — 24개 키포인트의 어휘와 라벨 읽기.

원본은 `ultralytics/cfg/datasets/dog-pose.yaml` 의 `kpt_names` 다. **여기서 순서를
바꾸면 안 된다** — 라벨 파일이 이 순서로 저장돼 있어서, 한 칸만 밀려도 코가 눈이 되고
목이 꼬리가 되는데 아무 데서도 에러가 안 난다.

라벨 한 줄의 생김새 (전부 정규화 0~1, 77개 필드):

    class  cx cy w h  x0 y0 v0  x1 y1 v1  ...  x23 y23 v23

`v` 는 0 이면 안 찍힌 것이고 그때 `x = y = 0` 이다. **`v` 를 안 보고 좌표만 쓰면
안 찍힌 키포인트가 전부 왼쪽 위 모서리(0, 0)로 몰려서 상자가 그쪽으로 늘어난다.**

이 파일은 **학습된 모델이 없어도 쓸 수 있다.** 정답 라벨로 얼굴 상자 계산을 먼저
검증하는 것이 이 실험의 0단계였고(`tools/verify_face_box.py`), 그때 쓰는 입구다.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image

from src.version import REPO_ROOT

# yaml 의 kpt_names 순서 그대로.
NAMES: tuple[str, ...] = (
    "front_left_paw",    # 0
    "front_left_knee",   # 1
    "front_left_elbow",  # 2
    "rear_left_paw",     # 3
    "rear_left_knee",    # 4
    "rear_left_elbow",   # 5
    "front_right_paw",   # 6
    "front_right_knee",  # 7
    "front_right_elbow", # 8
    "rear_right_paw",    # 9
    "rear_right_knee",   # 10
    "rear_right_elbow",  # 11
    "tail_start",        # 12
    "tail_end",          # 13
    "left_ear_base",     # 14
    "right_ear_base",    # 15
    "nose",              # 16
    "chin",              # 17
    "left_ear_tip",      # 18
    "right_ear_tip",     # 19
    "left_eye",          # 20
    "right_eye",         # 21
    "withers",           # 22  어깨 — 목 아래가 시작되는 자리
    "throat",            # 23  목
)

INDEX: dict[str, int] = {name: i for i, name in enumerate(NAMES)}

N_KEYPOINTS = len(NAMES)

# ---------------------------------------------------------------------------
# ⚠️ **선언된 24개 중 실제로 찍혀 있는 것은 20개다.**
#
# yaml 은 24개 이름을 적어 두지만, train 6,773장 · val 1,703장 전부를 세어 보니
# 아래 넷은 **한 장도 안 찍혀 있다** (`v` 가 언제나 0).
#
#     20 left_eye    0.0%      21 right_eye   0.0%
#     22 withers     0.0%      23 throat      0.0%
#
# dog-pose 의 뿌리인 StanfordExtra 가 24개 이름을 정의하고 그중 20개만 라벨링한
# 것을 그대로 물려받았다. **이걸 모르고 쓰면 조용히 망가진다** — 눈으로 얼굴
# 중심을 잡거나 `throat` 으로 목선을 정하는 코드는 예외 없이 폴백만 타면서도
# 에러는 안 낸다. 그래서 아래 [FACE] 는 **실제로 있는 6개만** 담는다.
#
# 각 키포인트가 찍혀 있는 비율 (val 기준):
#     nose 99.2% · chin 90.0% · left_ear_base 86.7% · right_ear_base 83.0%
#     left_ear_tip 67.4% · right_ear_tip 66.0%
# ---------------------------------------------------------------------------

#: 얼굴 상자를 만드는 데 쓰는 것. **실제로 라벨이 있는 6개뿐이다.**
#:
#: 눈이 빠진 것이 생각보다 손해가 아니다. 우리가 필요한 것은 눈동자 위치가 아니라
#: **머리의 경계**인데, 그건 귀끝(위)·코(앞)·턱(아래)이 눈보다 잘 잡는다.
FACE: tuple[int, ...] = (
    INDEX["left_ear_base"],
    INDEX["right_ear_base"],
    INDEX["nose"],
    INDEX["chin"],
    INDEX["left_ear_tip"],
    INDEX["right_ear_tip"],
)

#: 선언은 됐지만 이 데이터셋에 **라벨이 없는** 것들. 쓰면 안 된다.
#: 나중에 다른 데이터셋을 붙일 때 여기를 보고 판단하라고 남긴다.
UNLABELED: tuple[int, ...] = (
    INDEX["left_eye"],
    INDEX["right_eye"],
    INDEX["withers"],
    INDEX["throat"],
)

#: 목선용으로 **쓰려 했던** 자리. 지금은 둘 다 0.0% 라 못 쓴다.
#: `geometry.neck_t` 가 왜 턱과 귀밑으로 목을 추정하는지의 이유가 이것이다.
THROAT = INDEX["throat"]
WITHERS = INDEX["withers"]

#: 귀밑 한 쌍. 머리 꼭대기 쪽 기준선이라 머리 길이를 재는 자로 쓴다.
EAR_BASES = (INDEX["left_ear_base"], INDEX["right_ear_base"])
CHIN = INDEX["chin"]
NOSE = INDEX["nose"]

#: 얼굴 상자에 들면 "몸이 딸려 왔다" 고 볼 것들. 발·무릎·팔꿈치·꼬리.
#:
#: 팔꿈치(2·8)는 어깨에 가까워서 애매한데, **일부러 넣는다.** 팔꿈치가 얼굴 상자에
#: 들었다면 그 상자는 이미 가슴을 먹은 것이라 카드 구멍에서 머리가 밀린다.
BODY: tuple[int, ...] = tuple(
    INDEX[n]
    for n in (
        "front_left_paw", "front_left_knee", "front_left_elbow",
        "rear_left_paw", "rear_left_knee", "rear_left_elbow",
        "front_right_paw", "front_right_knee", "front_right_elbow",
        "rear_right_paw", "rear_right_knee", "rear_right_elbow",
        "tail_start", "tail_end",
    )
)

#: 좌우 뒤집기 증강을 쓸 때 짝이 되는 자리. 얼굴 9개만 따로 학습할 때 필요하다
#: (`tools/train_dog_pose.py` 의 갈래 B). 왼쪽↔오른쪽을 안 바꾸면 뒤집은 사진에서
#: 왼눈 자리에 오른눈 정답이 들어가 학습이 스스로 망가진다.
FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (INDEX["front_left_paw"], INDEX["front_right_paw"]),
    (INDEX["front_left_knee"], INDEX["front_right_knee"]),
    (INDEX["front_left_elbow"], INDEX["front_right_elbow"]),
    (INDEX["rear_left_paw"], INDEX["rear_right_paw"]),
    (INDEX["rear_left_knee"], INDEX["rear_right_knee"]),
    (INDEX["rear_left_elbow"], INDEX["rear_right_elbow"]),
    (INDEX["left_ear_base"], INDEX["right_ear_base"]),
    (INDEX["left_ear_tip"], INDEX["right_ear_tip"]),
    (INDEX["left_eye"], INDEX["right_eye"]),
)


# ---------------------------------------------------------------------------
# 라벨 읽기
#
# ## 라벨의 `cx cy w h` 는 믿지 않는다
#
# val 300장을 재 봤더니 **6.7% 에서 키포인트가 그 상자 밖으로 나갔다.** 몇 장
# (`n02085620_1152`)은 상자가 좌우로 뒤집힌 것처럼 반대편에 있다. 우리는 어차피
# 키포인트만 쓰므로 상자는 **정합성 검사용으로만** 읽어 두고, 넓이 같은 것은
# `facebox.body_box()` 로 키포인트에서 다시 만든다.
# ---------------------------------------------------------------------------


#: 데이터셋 기본 자리. `tools/fetch_dog_pose.py` 가 여기에 푼다.
#: git 밖이다 — 386MB 라 저장소에 두지 않는다.
DATASET_ROOT = REPO_ROOT / ".data" / "dog-pose"


@dataclass(frozen=True)
class Sample:
    stem: str
    image: pathlib.Path
    #: (24, 3) 정규화 `[x, y, v]`
    kps: np.ndarray
    #: 라벨 파일이 적어 둔 몸통 상자. 정규화 `[x0, y0, x1, y1]`. **믿지 않는다**
    label_box: tuple[float, float, float, float]

    _size: tuple[int, int] | None = None

    def size(self) -> tuple[int, int]:
        """`(width, height)` 픽셀. 헤더만 읽어서 화소는 안 푼다."""
        with Image.open(self.image) as im:
            return im.size

    def sane(self) -> bool:
        """라벨 상자와 키포인트가 서로 맞는가.

        어긋난 라벨을 지표에 섞으면 **우리 계산이 틀린 것인지 라벨이 틀린 것인지
        구분이 안 된다.** 그래서 세는 대신 걸러내고, 몇 장을 걸렀는지 따로 보고한다.
        """
        pts = self.kps[self.kps[:, 2] > 0][:, :2]
        if len(pts) == 0:
            return False
        x0, y0, x1, y1 = self.label_box
        # 라벨 상자를 5% 늘려 준다. 딱 맞게 재면 경계에 걸친 키포인트가 억울하게 걸린다.
        pad = 0.05
        w, h = x1 - x0, y1 - y0
        inside = (
            (pts[:, 0] >= x0 - w * pad)
            & (pts[:, 0] <= x1 + w * pad)
            & (pts[:, 1] >= y0 - h * pad)
            & (pts[:, 1] <= y1 + h * pad)
        )
        return bool(inside.all())


def parse_label(text: str) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """라벨 한 줄을 읽는다. 강아지가 여럿이면 **제일 큰 것** 하나만 쓴다.

    카드에는 한 마리만 들어간다. 여럿을 다 들고 있어 봐야 어느 것이 그 집 개인지는
    여기서 못 정한다 — 앱에서는 사용자가 상자로 고르는 자리다.
    """
    best = None
    for line in text.splitlines():
        f = line.split()
        if len(f) != 5 + N_KEYPOINTS * 3:
            continue
        v = [float(x) for x in f]
        area = v[3] * v[4]
        if best is None or area > best[0]:
            best = (area, v)
    if best is None:
        return None

    v = best[1]
    cx, cy, w, h = v[1:5]
    kps = np.asarray(v[5:], dtype=np.float32).reshape(N_KEYPOINTS, 3)
    return kps, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def load_split(split: str, root: pathlib.Path = DATASET_ROOT) -> Iterator[Sample]:
    """`train` 또는 `val` 을 stem 순으로 흘린다. 이미지는 아직 안 연다."""
    images = root / "images" / split
    labels = root / "labels" / split
    if not labels.is_dir():
        raise FileNotFoundError(
            f"{labels} 가 없습니다. 먼저 `uv run tools/fetch_dog_pose.py` 를 돌리세요."
        )

    for label_path in sorted(labels.glob("*.txt")):
        parsed = parse_label(label_path.read_text())
        if parsed is None:
            continue
        stem = label_path.stem
        image = images / f"{stem}.jpg"
        if not image.exists():
            continue
        kps, box = parsed
        yield Sample(stem=stem, image=image, kps=kps, label_box=box)
