import io

import numpy as np
import pytest
import soundfile as sf

from src.loop import (
    BLOCK,
    LoopError,
    crossfade_loop,
    read_audio,
    seam_jump,
    to_ogg,
    write_loop_ogg,
)
from src.music.mock import MockMusic

RATE = 44100


def sine(seconds: float, hz: float = 431.7, rate: int = RATE) -> np.ndarray:
    """루프 길이에 딱 안 떨어지는 주파수. 그냥 이으면 이음새가 튄다."""
    t = np.arange(int(rate * seconds)) / rate
    wave = np.sin(2 * np.pi * hz * t).astype(np.float32)
    return np.stack([wave, wave], axis=1)


def test_이음새가_안쪽_계단보다_크지_않다():
    """이게 '툭' 이 안 들리는 조건이다. 30초를 두 바퀴 들어야 사람 귀로도 확인된다."""
    raw = sine(6.0)
    loop = crossfade_loop(raw, RATE)
    inner = float(np.max(np.abs(np.diff(loop, axis=0))))
    assert seam_jump(loop) <= inner * 1.05
    assert seam_jump(loop) < seam_jump(raw)


def test_겹친_만큼_짧아진다():
    loop = crossfade_loop(sine(6.0), RATE, fade_seconds=1.5)
    assert len(loop) == int(RATE * 6.0) - int(RATE * 1.5)


def test_겹치는_구간에서_소리가_죽지_않는다():
    """직선으로 섞으면 겹친 구간이 0.7배로 한 번 꺼진다 — 등파워로 섞는 이유.

    서로 다른 소리끼리 섞을 때의 이야기라, 위상이 맞물리는 순음이 아니라 잡음으로
    잰다.
    """
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal((RATE * 6, 2)) * 0.3).astype(np.float32)
    loop = crossfade_loop(noise, RATE, fade_seconds=1.0)
    fade = int(RATE * 1.0)
    안쪽 = float(np.sqrt(np.mean(loop[fade: fade * 2] ** 2)))
    겹친곳 = float(np.sqrt(np.mean(loop[:fade] ** 2)))
    assert 0.9 < 겹친곳 / 안쪽 < 1.1


def test_섞다가_넘치면_되돌린다():
    loud = np.ones((RATE * 4, 2), dtype=np.float32) * 0.9
    assert float(np.max(np.abs(crossfade_loop(loud, RATE)))) <= 1.0


def test_너무_짧으면_그대로_둔다():
    short = sine(0.5)
    assert np.array_equal(crossfade_loop(short, RATE), short)


def test_OGG_로_쓰고_다시_읽을_수_있다(tmp_path):
    """MP3 는 이어 붙이면 29초마다 툭 소리가 나서 OGG 를 쓴다 (convert_audio.py)."""
    path = to_ogg(crossfade_loop(sine(3.0), RATE), RATE, tmp_path / "bgm.ogg")
    back, rate = sf.read(str(path), always_2d=True)
    assert path.suffix == ".ogg" and rate == RATE
    assert sf.info(str(path)).format == "OGG"
    assert back.shape[1] == 2


def test_블록으로_잘라_넘긴다(tmp_path, monkeypatch):
    """통째로 넘기면 libsndfile 이 윈도우에서 스택 오버플로로 조용히 죽는다."""
    long = np.zeros((BLOCK * 2 + 7, 2), dtype=np.float32)
    writes = []

    real = sf.SoundFile

    class Counting(real):
        def write(self, data):
            writes.append(len(data))
            return super().write(data)

    monkeypatch.setattr(sf, "SoundFile", Counting)
    to_ogg(long, RATE, tmp_path / "bgm.ogg")
    assert len(writes) == 3 and max(writes) == BLOCK


def test_provider_바이트에서_한_번에_OGG_까지(tmp_path):
    path = write_loop_ogg(MockMusic().generate("네옹", 4), tmp_path / "bgm.ogg")
    assert path.exists() and path.stat().st_size > 0


def test_읽을_수_없는_바이트는_곱게_실패한다():
    with pytest.raises(LoopError):
        read_audio("이건 음원이 아니다".encode("utf-8"))


def test_빈_음원도_곱게_실패한다():
    buf = io.BytesIO()
    sf.write(buf, np.zeros((0, 2), dtype=np.float32), RATE, format="WAV")
    with pytest.raises(LoopError):
        read_audio(buf.getvalue())
