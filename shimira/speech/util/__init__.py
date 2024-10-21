import numpy as np
from shimira.speech import constant as const
from shimira.speech.util import calc


class Audio:
    def __init__(self, data: np.ndarray, sr: int):
        self.__data = data
        self.__sr = sr

    def get_data(self) -> np.ndarray:
        return self.__data

    def get_sr(self) -> int:
        return self.__sr

    def normalize_rms(self, target_dbfs: float = -26) -> None:
        """信号のRMSを指定してノーマライズします

        Args:
            target_dbfs (float, optional): 目標 dBFS。波形のRMSがこの値になるように信号全体が定数倍されます。デフォルトは -26 dBFS。
        """
        _dbfs = calc.calc_rms_dbfs(self.__data)
        amp_const = 10 ** ((target_dbfs - _dbfs) / 20)
        self.__data = self.__data * amp_const

    def normalize_peak(self, target_abs_amp: float = 1) -> None:
        """信号の最大ピークを指定してノーマライズします

        Args:
            target_abs_amp (float, optional): 目標振幅。最大ピークの絶対値がこの値になるように信号全体が定数倍されます。デフォルトは1。
        """
        max_peak = np.max(np.abs(self.__data))
        amp_const = target_abs_amp / max_peak
        self.__data = self.__data * amp_const


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
