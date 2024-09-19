import numpy as np
from shimira.speech import constant as const


class Audio:
    def __init__(self, data: np.ndarray, sr: int):
        self.__data = data
        self.__sr = sr

    def get_data(self) -> np.ndarray:
        return self.__data

    def get_sr(self) -> int:
        return self.__sr


def get_FFT_hop_samples(sr: float) -> int:
    """FFTのhopサンプル数を算出します

    Args:
        sr (float): サンプリング周波数

    Returns:
        int: FFTのhopサンプル数
    """
    return int(const.FFT_HOP_MSEC * 0.001 * sr)


def get_FFT_window_samples() -> int:
    return const.FFT_WINDOW_SAMPLES
