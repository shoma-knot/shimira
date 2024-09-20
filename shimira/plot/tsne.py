import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional


def tSNE_2DEmbedding(data: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    return sklearn.manifold.TSNE(n_components=2, n_jobs=n_jobs).fit_transform(data)


def tSNE_2Dplot(
    data: list[np.ndarray], label: list[str], ax: Optional[Axes] = None, n_jobs: int = 1
) -> Axes:

    assert len(data) == len(label), "入力データ数とラベルの数が合いません"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    length_list = [d.shape[0] for d in data]
    concat_d = np.concatenate(data, axis=0, dtype=np.float32)

    embs = tSNE_2DEmbedding(data=concat_d, n_jobs=n_jobs)

    idx = 0
    for i, length in enumerate(length_list):
        ax.scatter(
            embs[idx : idx + length, 0], embs[idx : idx + length, 1], label=label[i]
        )
        idx += length

    return ax
