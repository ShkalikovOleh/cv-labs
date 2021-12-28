from itertools import product
from typing import Callable, Tuple

import numpy as np

from .GrammarBase import GrammarBase


def __get_rectangles_from_starting_point(shape: np.ndarray, p: Tuple, S: int):
    ndim = len(shape)
    t = np.minimum(1+S, shape - p)
    indexes = product(*[np.arange(t[i]) + p[i] for i in range(ndim)])
    return filter(lambda idx: np.prod(np.subtract(idx, p) + 1) == S, indexes)


def __get_rectangles(shape: np.ndarray, S: int):
    for p in product(*[range(h) for h in shape]):
        for e in __get_rectangles_from_starting_point(shape, p, S):
            yield p + e


def __get_splitting_idx(idx: Tuple[int, ...], k: int, ndim: int,
                        split: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    idx1t = list(idx)
    idx1t[k + ndim] = split
    idx1 = tuple(idx1t)

    idx2t = list(idx)
    idx2t[k] = split + 1
    idx2 = tuple(idx2t)

    return idx1, idx2


def __mult_qform(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.einsum('i, aik, k->a', a, b, c)


def __or_reduction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logical_or(a, b)


qform_func = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
reduction_func = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cyk(input, grammar: GrammarBase,
        reduction: reduction_func = __or_reduction,
        qform: qform_func = __mult_qform) -> np.ndarray:

    F, shape = grammar.recognize_terminal(input)
    ndim = grammar.ndim
    max_S = np.prod(shape)

    if max_S < 2:
        return F[tuple(0 for _ in range(ndim*2))]

    for s in range(2, max_S + 1):
        for idx in __get_rectangles(shape, s):
            for k in range(ndim):
                g = grammar.get_rules_for_dim(k)
                for split in range(idx[k], idx[k + ndim]):
                    idx1, idx2 = __get_splitting_idx(idx, k, ndim, split)
                    res = qform(F[idx1], g, F[idx2])
                    F[idx] = reduction(F[idx], res)

    return F[idx]
