from functools import partial
from itertools import combinations
from typing import Callable

import numpy as np


def g(images: np.ndarray, masks: np.ndarray, b: float, ord: int = 1):
    N, H, W, _ = images.shape
    # float32 for speed up computation
    G = np.zeros((N, N, H, W-1), np.float32)

    for i, j in combinations(range(N), 2):
        gij = np.linalg.norm(images[i] - images[j], ord=ord, axis=2)
        G[i, j] = G[j, i] = gij[:, :-1] + gij[:, 1:]  # g is symmetric

    return b * G


def q(images: np.ndarray, masks: np.ndarray, a: float):
    return a * (1 - masks)


_g = partial(g, b=0.1)
_q = partial(q, a=100)
__weight_func = Callable[[np.ndarray, np.ndarray], np.ndarray]


def merge(images: np.ndarray, masks: np.ndarray,
          g: __weight_func = _g, q: __weight_func = _q):

    N, H, W, _ = images.shape

    Q = q(images, masks)
    G = g(images, masks)

    # for speed up computation width is the first axis
    F = np.empty((W, N, H), np.float32)
    F[-1] = np.min(Q[:, :, -1] + G[:, :, :, -1], axis=1)
    for i in range(W-2, 0, -1):
        F[i-1] = np.min(Q[:, :, i] + F[i] + G[:, :, :, i], axis=1)

    IDX = np.empty((W, H), np.int32)
    IDX[0] = np.argmin(Q[:, :, 0] + F[0], axis=0)
    for i in range(1, W):
        g = np.take_along_axis(G[:, :, :, i-1],
                               IDX[i-1, np.newaxis, np.newaxis, :], axis=0)[0]
        sum = Q[:, :, i] + g + F[i]
        IDX[i] = np.argmin(sum, axis=0)

    return np.take_along_axis(images, IDX.T[np.newaxis, :, :, np.newaxis], axis=0)[0]
