"""루프 이음새와 OGG 변환.

**왜 OGG 인가** — MP3 는 인코더가 앞뒤에 빈 프레임을 붙이기 때문에 이어 붙이면
29초쯤마다 "툭" 소리가 난다. `DAENGS_APP/tools/convert_audio.py` 가 MP3→OGG 를 하는
이유가 그것이고, 여기서도 규격을 그대로 따른다.

**왜 블록으로 넘기는가** — libsndfile 에 통째로 넘기면 윈도우에서 스택 오버플로로
조용히 죽는다. `BLOCK = 1<<16` 씩 잘라 넘기는 것까지 그대로 가져왔다.

이음새는 **꼬리를 머리에 겹쳐** 만든다. 겹치는 구간의 시작이 원래 그 앞에 있던
소리라, 마지막 샘플에서 첫 샘플로 넘어갈 때 값이 튀지 않는다.
"""

import io
from pathlib import Path

import numpy as np
import soundfile as sf

# convert_audio.py 와 같은 값. 이유는 모듈 설명 참고.
BLOCK = 1 << 16

DEFAULT_FADE_SECONDS = 1.5


class LoopError(RuntimeError):
    """음원을 읽거나 쓰지 못했다. 카드는 그대로 나가야 한다."""


def read_audio(data: bytes) -> tuple[np.ndarray, int]:
    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise LoopError(f"음원을 읽지 못했다: {exc}") from exc
    if samples.size == 0:
        raise LoopError("음원이 비었다")
    return samples, rate


def crossfade_loop(samples: np.ndarray, rate: int,
                   fade_seconds: float = DEFAULT_FADE_SECONDS) -> np.ndarray:
    """끝에서 처음으로 돌아갈 때 안 튀는 길이로 다시 만든다.

    꼬리 `fade` 만큼을 머리에 겹쳐 섞고 꼬리는 버린다. 결과 길이는
    원래 길이 - fade 다.
    """
    if samples.ndim != 2:
        raise LoopError("2차원 (프레임, 채널) 배열이어야 한다")
    length = len(samples)
    fade = int(rate * fade_seconds)
    if fade <= 0 or length <= fade * 2:
        # 겹칠 자리가 없다. 이음새를 못 만드니 그대로 둔다 — 짧은 소리는 어차피
        # 루프로 안 쓴다.
        return samples.astype(np.float32, copy=True)

    head = samples[:fade].astype(np.float64)
    tail = samples[length - fade:].astype(np.float64)

    # 등파워 크로스페이드. 직선으로 섞으면 겹치는 구간에서 소리가 한 번 죽는다.
    ramp = np.linspace(0.0, np.pi / 2, fade, dtype=np.float64)[:, None]
    mixed = tail * np.cos(ramp) + head * np.sin(ramp)

    out = samples[: length - fade].astype(np.float64).copy()
    out[:fade] = mixed
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:                      # 섞다가 넘친 것만 되돌린다
        out /= peak
    return out.astype(np.float32)


def to_ogg(samples: np.ndarray, rate: int, path: Path) -> Path:
    """OGG/Vorbis 로 쓴다. 통째로 넘기지 않고 블록으로 잘라 넘긴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sf.SoundFile(str(path), mode="w", samplerate=rate,
                          channels=samples.shape[1], format="OGG",
                          subtype="VORBIS") as out:
            for start in range(0, len(samples), BLOCK):
                out.write(samples[start:start + BLOCK])
    except Exception as exc:
        raise LoopError(f"OGG 로 쓰지 못했다: {exc}") from exc
    return path


def write_loop_ogg(data: bytes, path: Path,
                   fade_seconds: float = DEFAULT_FADE_SECONDS) -> Path:
    """provider 가 준 음원 바이트를 루프용 OGG 한 개로 만든다."""
    samples, rate = read_audio(data)
    return to_ogg(crossfade_loop(samples, rate, fade_seconds), rate, path)


def seam_jump(samples: np.ndarray) -> float:
    """마지막 샘플에서 첫 샘플로 넘어갈 때 값이 얼마나 튀는지. 클수록 '툭' 이 들린다."""
    if len(samples) < 2:
        return 0.0
    return float(np.max(np.abs(samples[0] - samples[-1])))
