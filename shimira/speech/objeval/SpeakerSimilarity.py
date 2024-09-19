import numpy as np
import pathlib
import enum


class VoiceEncoderModel(enum.Enum):
    Resemblyzer = enum.auto()

    def __init__(self):
        self.__model = None
        self.__preprocess_func = None

    def load(self):
        match self:
            case VoiceEncoderModel.Resemblyzer:
                import resemblyzer
                self.__model = resemblyzer.VoiceEncoder()
                self.__preprocess_func = resemblyzer.audio.preprocess_wav
            case _:
                raise NotImplementedError()

    def encode(self, wav: str | pathlib.Path | np.ndarray) -> np.ndarray:
        match self:
            case VoiceEncoderModel.Resemblyzer:
                preprocess_wav = self.__preprocess_func(wav)
                return self.__model.embed_utterance(preprocess_wav)
            case _:
                raise NotImplementedError()


def calc_speaker_similarity_from_waveform(wave_a: np.ndarray, wave_b: np.ndarray, encoder: VoiceEncoderModel) -> float:
    """話者埋め込みモデルを用いて、入力された音声波形同士の話者類似度を算出します。
    :param wave_a: 話者類似度を計算したい音声Aの波形が保存された np.ndarray。shapeは [1, T] でなければならない。
    :param wave_b: 話者類似度を計算したい音声Bの波形が保存された np.ndarray。shapeは [1, T] でなければならない。
    :param encoder:　VoiceEncoderModelを用いて指定された、話者埋め込みモデル。
    :return: 音声Aと音声Bの話者類似度。
    """

    if len(wave_a.shape) != 2 or wave_a.shape[0] != 1:
        raise ValueError(f"入力波形Aのshapeが変です: [1, T] が予期されましたが、{wave_a.shape} が入力されました")
    if len(wave_b.shape) != 2 or wave_b.shape[0] != 1:
        raise ValueError(f"入力波形Bのshapeが変です: [1, T] が予期されましたが、{wave_b.shape} が入力されました")

    embed_a = encoder.encode(wave_a)
    embed_b = encoder.encode(wave_b)
    return np.inner(embed_a, embed_b)


def calc_speaker_similarity_from_file(fpath_a: str | pathlib.Path, fpath_b: str | pathlib.Path,
                                      encoder: VoiceEncoderModel) -> float:
    """話者埋め込みモデルを用いて、入力された音声ファイル同士の話者類似度を算出します。
    :param fpath_a: 話者類似度を計算したい音声Aのファイル。str か pathlib.Path でなければならない。
    :param fpath_b: 話者類似度を計算したい音声Bのファイル。str か pathlib.Path でなければならない。
    :param encoder:　VoiceEncoderModelを用いて指定された、話者埋め込みモデル。
    :return: 音声Aと音声Bの話者類似度。
    """

    if isinstance(fpath_a, (str, pathlib.Path)):
        raise ValueError(
            f"引数 \"fpath_a\" の型が変です: str か pathlib.Path が予期されましたが、{type(fpath_a)} が入力されました")
    if isinstance(fpath_b, (str, pathlib.Path)):
        raise ValueError(
            f"引数 \"fpath_b\" の型が変です: str か pathlib.Path が予期されましたが、{type(fpath_b)} が入力されました")
    
    embed_a = encoder.encode(fpath_a)
    embed_b = encoder.encode(fpath_b)

    return np.inner(embed_a, embed_b)
    
    