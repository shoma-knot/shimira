import numpy as np


def calc_rms_dbfs(x: np.ndarray) -> float:
    """信号のRMSをdBFSで計算します。

    Args:
        x (np.ndarray): 計算対象の信号

    Returns:
        float: 信号のRMS。単位は dBFS。
    """
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x)))))
