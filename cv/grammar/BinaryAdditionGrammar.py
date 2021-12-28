from itertools import product
from typing import Tuple

import numpy as np

from .GrammarBase import GrammarBase


class BinaryAdditionGrammar(GrammarBase):

    def __init__(self, zero_mask: np.ndarray, one_mask: np.ndarray, threshold: float = 0.75) -> None:
        assert 0 < threshold <= 1
        assert one_mask.ndim == 2
        assert one_mask.shape == zero_mask.shape

        super().__init__()

        self.__one_mask = one_mask
        self.__zero_mask = zero_mask
        self.__threshold = threshold

        self.__rules = np.zeros((2, 10, 10, 10), dtype=bool)
        self.__rules[1, 0, 0, 0] = True  # I -> I | I
        self.__rules[1, 0, 2, 1] = True  # I -> v1 | I'
        self.__rules[1, 1, 3, 1] = True  # I' -> v1' | I'
        self.__rules[1, 1, 1, 0] = True  # I' -> I' | I
        self.__rules[0, 0, 6, 4] = True  # I -> A00/R0
        self.__rules[0, 0, 7, 5] = True  # I -> A01/R1
        self.__rules[0, 0, 8, 5] = True  # I -> A10/R1
        self.__rules[0, 1, 9, 4] = True  # I' -> A11/R0
        self.__rules[0, 2, 6, 5] = True  # v1 -> A00/R1
        self.__rules[0, 3, 7, 4] = True  # v1' -> A01/R0
        self.__rules[0, 3, 8, 4] = True  # v1' -> A10/R0
        self.__rules[0, 3, 9, 5] = True  # v1' -> A11/R1
        self.__rules[0, 6, 4, 4] = True  # A00 -> R0/R0
        self.__rules[0, 7, 4, 5] = True  # A01 -> R0/R1
        self.__rules[0, 8, 5, 4] = True  # A10 -> R1/R0
        self.__rules[0, 9, 5, 5] = True  # A11 -> R1/R1

    @property
    def ndim(self) -> int:
        return 2

    def __recognize(self, mask: np.ndarray, image_part: np.ndarray) -> bool:
        res = np.logical_and(mask, image_part)
        return (res.sum() / mask.sum()) > self.__threshold

    def recognize_terminal(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = input.shape
        mH, mW = self.__one_mask.shape

        h = H // mH
        w = W // mW

        F = np.zeros((h, w, h, w, 10), dtype=bool)

        for i, j in product(range(h), range(w)):
            image_part = input[i*mH: (i+1)*mH, j*mW: (j+1)*mW]

            if self.__recognize(self.__one_mask, image_part):
                F[i, j, i, j, 5] = True
            elif self.__recognize(self.__zero_mask, image_part):
                F[i, j, i, j, 4] = True

        return F, np.asarray([h, w])

    def get_rules_for_dim(self, dim: int) -> np.ndarray:
        return self.__rules[dim]
