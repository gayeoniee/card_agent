"""mock provider — 네트워크·비용 없이 파이프라인 전체를 돌린다.

이게 있어야 음악 서비스가 정해지기 전에도 ①~⑦ 이 한 번에 도는지 볼 수 있고,
테스트가 유료 API 를 부르지 않는다.
"""

import io
import zlib

import numpy as np
import soundfile as sf

from src.music.base import MusicUnavailable

SAMPLE_RATE = 44100
# 5음계(펜타토닉)에서 고른 화음. 어긋난 음이 안 나오니 mock 으로 듣기에 낫다.
CHORD_HZ = (196.00, 293.66, 392.00, 587.33)


class MockMusic:
    """같은 프롬프트면 같은 소리가 나온다 (테스트가 흔들리지 않게)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def generate(self, prompt: str, seconds: int) -> bytes:
        if seconds <= 0:
            raise MusicUnavailable(f"길이가 0 이하다: {seconds}")

        n = int(self.sample_rate * seconds)
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        # 파이썬의 hash() 는 프로세스마다 씨가 달라서(PYTHONHASHSEED) 같은 프롬프트로도
        # 매번 다른 소리가 난다. 프로세스를 넘어 같은 값이 나오는 crc32 를 쓴다.
        rng = np.random.default_rng(zlib.crc32(prompt.encode("utf-8")))

        wave = np.zeros(n, dtype=np.float64)
        for i, hz in enumerate(CHORD_HZ):
            # 음마다 다른 느린 흔들림. 그래야 한 음처럼 안 들린다.
            lfo = 1.0 + 0.03 * np.sin(2 * np.pi * (0.05 + 0.02 * i) * t + rng.random() * 6.28)
            wave += np.sin(2 * np.pi * hz * t) * lfo / len(CHORD_HZ)

        # 앞뒤를 재우지 않는다 — 루프 이음새는 src/loop.py 가 만든다.
        wave *= 0.35
        stereo = np.stack([wave, np.roll(wave, self.sample_rate // 200)], axis=1)

        buf = io.BytesIO()
        sf.write(buf, stereo, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


class FailingMusic:
    """실패 경로를 보려고 두는 provider. bgm 이 null 이어도 카드는 나와야 한다."""

    def generate(self, prompt: str, seconds: int) -> bytes:
        raise MusicUnavailable("mock 실패 provider")
