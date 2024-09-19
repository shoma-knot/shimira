import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional


def tSNE_2Dplot(data: np.ndarray, ax: Optional[Axes] = None, n_jobs: int = 1) -> Axes:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    embs = sklearn.manifold.TSNE(n_components=2, n_jobs=n_jobs).fit_transform(data)

    ax.scatter(embs[:, 0], embs[:, 1])
