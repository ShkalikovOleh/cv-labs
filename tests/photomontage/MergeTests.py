from functools import partial
import unittest

import numpy as np

from cv.photomontage import q, g, merge


class MergeTests(unittest.TestCase):

    def test_q(self):
        masks = np.array([[[1, 0],
                          [0, 0]],
                          [[0, 0],
                          [0, 1]]])
        images = np.empty((2, 2, 2, 3))

        Q = q(images, masks, 1)

        expected_Q = 1 - masks
        self.assertEqual(expected_Q.shape, Q.shape)
        self.assertTrue(np.allclose(Q, expected_Q))

    def test_g(self):
        masks = np.empty((2, 2, 2))
        image1 = np.repeat(range(4), 3, axis=0).reshape(2, 2, 3)
        image2 = np.roll(image1, 1, axis=1)
        images = np.stack([image1, image2])

        G = g(images, masks, b=1)

        expected_G = np.array([0, 0, 6, 6, 6, 6, 0, 0]).reshape(2, 2, 2, 1)
        self.assertEqual(G.shape, expected_G.shape)
        self.assertTrue(np.allclose(G, expected_G))

    def test_merge(self):
        image1 = np.repeat(range(4), 3, axis=0).reshape(2, 2, 3)
        image2 = np.roll(image1, 1, axis=1)
        images = np.stack([image1, image2])

        masks = np.array([[[1, 0],
                           [0, 0]],
                          [[0, 0],
                           [0, 1]]])

        fq = partial(q, a=10)
        fg = partial(g, b=1)
        result = merge(images, masks, fg, fq)

        expected_result = image1
        expected_result[1, 1, :] = 2
        self.assertEqual(expected_result.shape, result.shape)
        self.assertTrue(np.allclose(result, expected_result))
