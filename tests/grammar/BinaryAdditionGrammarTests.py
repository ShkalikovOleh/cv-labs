import unittest

import numpy as np

from cv.grammar import BinaryAdditionGrammar


class BinaryAdditionGrammarTests(unittest.TestCase):

    def test_recognize(self):
        one_mask = np.array([[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]])
        zero_mask = np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])

        grammar = BinaryAdditionGrammar(zero_mask, one_mask)

        F, shape = grammar.recognize_terminal(np.hstack([one_mask, zero_mask]))

        self.assertEqual(shape[0], 1)
        self.assertEqual(shape[1], 2)

        self.assertTrue(F[0, 0, 0, 0, 5])
        self.assertFalse(F[0, 0, 0, 0, 4])
        self.assertTrue(F[0, 1, 0, 1, 4])
        self.assertFalse(F[0, 1, 0, 1, 5])

    def test_threshold(self):
        one_mask = np.array([[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]])
        zero_mask = np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])

        one_input = one_mask.copy()
        one_input[0, 0] = True

        grammar = BinaryAdditionGrammar(zero_mask, one_mask, threshold=1)

        F, shape = grammar.recognize_terminal(np.hstack([one_input, zero_mask]))

        self.assertFalse(F[0, 0, 0, 0, 5])
        self.assertFalse(F[0, 0, 0, 0, 4])
