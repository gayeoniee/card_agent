"""털색 — 누끼 안쪽만 보고 accent · accent2 를 뽑는다.

이머시브의 먼지·이슬·빛 색이 강아지 털색을 따라가면 장면이 그 강아지 것이 된다.
배경까지 세면 벽지 색이 뽑히므로 **알파 안쪽만** 센다.

accent2 는 accent 를 밝은 쪽으로 섞은 값이다. 두 번째 군집을 쓰면 강아지에 따라
보색이 튀어나와 장면이 요란해진다 — 앱의 배추 장면(#EEBE93 · #F5D9BE)도 같은
색의 밝은 짝이다.
"""

import colorsys
from dataclasses import dataclass

import numpy as np
from PIL import Image

# 군집 수. 흰 배와 검은 코까지 갈라 놓고 그중 하나를 고르려는 것이지, 색을 다
# 쓰려는 게 아니다.
CLUSTERS = 5
# 색을 셀 때 뽑아 보는 픽셀 수. 사진이 커도 여기서 잘린다 (속도).
SAMPLE = 20000
# 이 아래로 어둡거나 이 위로 밝으면 빛 색으로 못 쓴다. 검은 개의 검정을 그대로
# 쓰면 이머시브에서 아무것도 안 보인다.
MIN_V, MAX_V = 0.28, 0.94
MIN_S = 0.10
# accent2 를 얼마나 밝은 쪽으로 섞을지.
LIGHTEN = 0.35


@dataclass(frozen=True)
class Coat:
    accent: str
    accent2: str
    palette: list[tuple[str, float]]      # (hex, 차지하는 비율) — 큰 것부터


def to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, int(round(v)))) for v in rgb))


def lighten(rgb: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return tuple(c + (255.0 - c) * amount for c in rgb)


def _kmeans(pixels: np.ndarray, k: int, *, seed: int = 0, iters: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """작은 k-means. 씨를 고정해서 같은 사진이면 같은 색이 나오게 한다."""
    rng = np.random.default_rng(seed)
    k = min(k, len(pixels))
    # k-means++ 로 첫 자리를 잡는다. 무작위로 잡으면 군집이 붙어 버려 색이 뭉갠다.
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(k - 1):
        d = np.min(((pixels[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d.sum()
        probs = d / total if total > 0 else np.full(len(pixels), 1 / len(pixels))
        centers.append(pixels[rng.choice(len(pixels), p=probs)])
    centers = np.array(centers, dtype=np.float64)

    labels = np.zeros(len(pixels), dtype=np.int64)
    for _ in range(iters):
        dist = ((pixels[:, None, :] - centers[None]) ** 2).sum(-1)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            hit = pixels[labels == i]
            if len(hit):
                centers[i] = hit.mean(axis=0)
    return centers, labels


def _usable(rgb: np.ndarray) -> bool:
    r, g, b = (float(c) / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return MIN_V <= v <= MAX_V and s >= MIN_S


def _pull_into_range(rgb: np.ndarray) -> tuple[float, float, float]:
    """빛으로 쓸 수 있는 범위로 끌어당긴다. 색상(hue)은 건드리지 않는다."""
    r, g, b = (float(c) / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = min(max(v, MIN_V), MAX_V)
    s = max(s, MIN_S)
    return tuple(c * 255 for c in colorsys.hsv_to_rgb(h, s, v))


def coat_colors(img: Image.Image, *, seed: int = 0, alpha_min: int = 200) -> Coat:
    """누끼 한 장에서 accent · accent2 를 뽑는다."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
    inside = arr[arr[:, 3] >= alpha_min][:, :3]
    if len(inside) == 0:
        raise ValueError("알파 안쪽에 픽셀이 없다 — 누끼가 비었다")

    rng = np.random.default_rng(seed)
    if len(inside) > SAMPLE:
        inside = inside[rng.choice(len(inside), SAMPLE, replace=False)]

    centers, labels = _kmeans(inside.astype(np.float64), CLUSTERS, seed=seed)
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(-counts)

    palette = [
        (to_hex(tuple(centers[i])), round(float(counts[i]) / len(inside), 4))
        for i in order
        if counts[i] > 0
    ]

    # 큰 군집부터 보되, 빛으로 못 쓸 색(너무 어둡거나 회색)은 건너뛴다.
    # 다 못 쓰면 제일 큰 것을 범위 안으로 끌어당겨 쓴다 — 색을 못 정하는 것보다 낫다.
    chosen = next((centers[i] for i in order if counts[i] > 0 and _usable(centers[i])), None)
    base = _pull_into_range(centers[order[0]] if chosen is None else chosen)

    return Coat(accent=to_hex(base), accent2=to_hex(lighten(base, LIGHTEN)), palette=palette)
