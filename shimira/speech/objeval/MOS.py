import numpy as np
from pathlib import Path
import torch
import librosa
import enum


class MOSPredictModel(enum.Enum):
    UTMOS = enum.auto()

    def __init__(self):
        self.__model = None

    def predict(self, *args, **kwargs):
        if self.__model is None:
            self.load()
        return self.__model(*args)

    def load(self):
        match self:
            case MOSPredictModel.UTMOS:
                self.__model = torch.hub.load(
                    "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
                )
            case _:
                raise NotImplementedError(f"モデル {self.name} は未実装です。")


def calc_predicted_mos_from_waveform(
    wave: np.ndarray, sr: float, predictor: MOSPredictModel
) -> float:
    """自然性MOSを予測するモデルを用いて、入力された音声波形の自然性予測MOS値を算出します。

    :param wave: 音声波形の numpy.ndarray。shapeは [1, T] でなければならない。
    :param sr: 音声波形のサンプリング周波数。
    :param predictor: MOSPredictModelを用いて指定された、自然性予測MOS値を算出するモデル。
    :return: 入力された音声波形の自然性予測MOS値。
    """
    wave_tensor = torch.from_numpy(wave)

    if len(wave_tensor.shape) != 2 or wave_tensor.shape[0] != 1:
        raise ValueError(
            f"入力波形のshapeが変です: [1, T] が予期されましたが、{wave_tensor} が入力されました"
        )

    return predictor.predict(wave_tensor, sr).detach().cpu().numpy()[0]


def calc_predicted_mos_from_file(fpath: Path | str, predictor: MOSPredictModel):
    """自然性MOSを予測するモデルを用いて、入力された音声ファイルの自然性予測MOS値を算出します。

    :param fpath: 音声ファイルのパス。
    :param predictor: MOSPredictModelを用いて指定された、自然性予測MOS値を算出するモデル。
    :return: 入力された音声波形の自然性予測MOS値。
    """

    if isinstance(fpath, (str, Path)):
        raise ValueError(
            f'引数 "fpath" の型が変です: str か pathlib.Path が予期されましたが、{type(fpath)} が入力されました'
        )

    if isinstance(fpath, str):
        fpath = Path(fpath)

    if not fpath.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {fpath}")

    wave, sr = librosa.load(fpath, sr=None, mono=True)
    wave_tensor = torch.from_numpy(wave).unsqueeze(0)

    if len(wave_tensor.shape) != 2 or wave_tensor.shape[0] != 1:
        raise ValueError(
            f"入力波形のshapeが変です: [1, T] が予期されましたが、{wave_tensor} が入力されました"
        )

    return predictor.predict(wave_tensor, sr).detach().cpu().numpy()[0]
