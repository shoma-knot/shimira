import numpy as np
import pyworld as pw
import parselmouth as pm
import librosa as lr

from shimira.speech import util
from shimira.speech.util import Audio


def get_f0(audio: Audio) -> np.ndarray:
    """harvest を用いて f0 を算出します

    Args:
        audio (Audio): 入力音声

    Returns:
        np.ndarray: 入力音声の f0
    """

    f0, t = pw.harvest(audio.get_data(), audio.get_sr())
    return f0


def get_uv(audio: Audio) -> np.ndarray:
    """dio を用いて 無声/有声区間 を算出します


    Args:
        audio (Audio): 入力音声

    Returns:
        np.ndarray: 入力音声の uvフラグ (true: 有声, false: 無声)
    """

    f0, _ = pw.dio(audio.get_data(), audio.get_sr())

    return f0 != 0.0


def get_formant(audio: Audio) -> np.ndarray:
    """praat の実装を用いてフォルマントを算出します

    Args:
        audio (Audio): 入力音声

    Returns:
        np.ndarray: 入力音声のフォルマント
    """
    snd = pm.Sound(samples=audio.get_data(), sampling_frequency=audio.get_sr())
    formant = snd.to_formant_burg()
    return np.array(
        [
            [formant.get_value_at_time(i, x, "HERTZ") for x in formant.xs()]
            for i in range(1, 5)
        ]
    )


def get_spectral_centroid(audio: Audio) -> np.ndarray:
    """librosa の実装を用いて、入力音声のスペクトル重心を算出します

    Args:
        audio (Audio): 入力音声

    Returns:
        np.ndarray: 入力音声のスペクトル重心
    """

    return lr.feature.spectral_centroid(
        y=audio.get_data(),
        sr=audio.get_sr(),
        n_fft=util.get_FFT_window_samples(),
        hop_length=util.get_FFT_hop_samples(),
    )
