import io

import numpy as np
import pytest
import soundfile as sf

from src.music.base import MusicProvider, MusicUnavailable, prompt_for, save_bytes
from src.music.mock import FailingMusic, MockMusic


def test_mock_은_네트워크_없이_소리를_만든다():
    data = MockMusic().generate("테스트", 2)
    samples, rate = sf.read(io.BytesIO(data), always_2d=True)
    assert rate == 44100
    assert samples.shape == (88200, 2)
    assert np.max(np.abs(samples)) > 0.05


def test_같은_프롬프트면_같은_소리다():
    music = MockMusic()
    assert music.generate("네옹", 1) == music.generate("네옹", 1)


def test_길이가_0_이면_거절한다():
    with pytest.raises(MusicUnavailable):
        MockMusic().generate("네옹", 0)


def test_인터페이스는_generate_하나뿐이다():
    assert isinstance(MockMusic(), MusicProvider)
    assert isinstance(FailingMusic(), MusicProvider)


def test_실패_provider_는_MusicUnavailable_을_올린다():
    with pytest.raises(MusicUnavailable):
        FailingMusic().generate("네옹", 30)


def test_프롬프트에_작물과_이름이_들어간다():
    prompt = prompt_for("단호박", "네옹")
    assert "단호박" in prompt and "네옹" in prompt


def test_바이트를_그대로_저장한다(tmp_path):
    path = save_bytes(b"1234", tmp_path / "a" / "bgm.wav")
    assert path.read_bytes() == b"1234"
