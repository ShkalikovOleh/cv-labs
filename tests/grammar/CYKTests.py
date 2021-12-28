import unittest

import numpy as np

from cv.grammar import cyk, GrammarBase


class Test1DGrammar(GrammarBase):

    def __init__(self) -> None:
        super().__init__()
        self.__rules = np.zeros((3, 3, 3), dtype=bool)
        self.__rules[0, 0, 0] = True
        self.__rules[0, 1, 2] = True

    def recognize_terminal(self, input):
        n = input.shape[0]
        F = np.zeros((n, n, 3), dtype=bool)

        idx0 = np.argwhere(input == '0')
        idx1 = np.argwhere(input == '1')
        F[idx0, idx0, 1] = True
        F[idx1, idx1, 2] = True

        return F, np.asarray([n])

    @property
    def ndim(self):
        return 1

    def get_rules_for_dim(self, ndim):
        return self.__rules


class Test2DGrammar(GrammarBase):

    def __init__(self) -> None:
        super().__init__()
        self.__rules = np.zeros((2, 3, 3, 3), dtype=bool)
        self.__rules[0, 0, 0, 0] = True
        self.__rules[1, 0, 0, 0] = True
        self.__rules[1, 0, 1, 2] = True

    def recognize_terminal(self, input):
        H, W = input.shape
        F = np.zeros((H, W, H, W, 3), dtype=bool)

        idx0 = np.argwhere(input == '0')
        idx1 = np.argwhere(input == '1')
        F[idx0[:, 0], idx0[:, 1], idx0[:, 0], idx0[:, 1], 1] = True
        F[idx1[:, 0], idx1[:, 1], idx1[:, 0], idx1[:, 1], 2] = True

        return F, np.asarray([H, W])

    @property
    def ndim(self):
        return 2

    def get_rules_for_dim(self, ndim):
        return self.__rules[ndim]


class CYKTests(unittest.TestCase):

    def test_1D_cyk(self):
        grammar = Test1DGrammar()
        input = np.asarray(list('0101010101'))

        res = cyk(input, grammar)

        self.assertTrue(res[0])

    def test_2D_cyk(self):
        grammar = Test2DGrammar()
        input_row = np.asarray(list('0101010101'))
        input = np.vstack([input_row, input_row, input_row])

        res = cyk(input, grammar)

        self.assertTrue(res[0])

    def test_cyk_2D_should_be_false(self):
        grammar = Test2DGrammar()
        input_row1 = np.asarray(list('01010101010'))
        input_row2 = np.asarray(list('0101011101'))
        input_row3 = np.asarray(list('1010101010'))
        input1 = np.vstack([input_row1, input_row1, input_row1])
        input2 = np.vstack([input_row2, input_row2, input_row2])
        input3 = np.vstack([input_row3, input_row3, input_row3])
        input4 = np.vstack([input_row2, input_row2, input_row3])

        res1 = cyk(input1, grammar)
        res2 = cyk(input2, grammar)
        res3 = cyk(input3, grammar)
        res4 = cyk(input4, grammar)

        self.assertFalse(res1[0])
        self.assertFalse(res2[0])
        self.assertFalse(res3[0])
        self.assertFalse(res4[0])
