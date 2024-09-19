from pathlib import Path
from typing import Optional
from macha.speech.util import Audio
import librosa as lr
import soundfile as sf


def load_wav(path: Path | str, sr: Optional[int] = None) -> Audio:
    data, sr = lr.load(path, sr=sr)
    return Audio(data, sr)


def save_wav(path: Path | str, audio: Audio) -> None:
    sf.write(path, audio.get_data(), audio.get_sr(), subtype="FLOAT")
