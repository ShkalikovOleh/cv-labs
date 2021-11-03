import unittest

from jax import numpy as jnp

from cv.supervised import GaussPerceptron


class GaussPerceptronTests(unittest.TestCase):

    def test_imax_nter_should_be_greater_0(self):
        with self.assertRaises(AssertionError):
            gp = GaussPerceptron(-1)
        with self.assertRaises(AssertionError):
            gp = GaussPerceptron(0)

    def test_should_be_raise_if_not_fitted(self):
        gp = GaussPerceptron()

        with self.assertRaises(ValueError):
            gp.predict(jnp.ones(2))

    def test_transform_X(self):
        gp = GaussPerceptron()
        X = jnp.array([[1, 2]])

        T = gp._transform_X(X)

        expected_T = jnp.array([[1, 2, 2, 4, 1, 2, 1]])
        self.assertTrue(jnp.allclose(T, expected_T))

    def test_fit(self):
        gp = GaussPerceptron(max_niter=100)
        X = jnp.array([[0.5, 3],
                       [-1, -2]])
        Y = jnp.array([1, -1])

        gp.fit(X, Y)

        self.assertEqual((Y * (gp._transform_X(X) @ gp.w) > 0).sum(), 2)

    def test_predict(self):
        gp = GaussPerceptron()
        X = jnp.array([[0.5, 3],
                       [-1, -2]])
        Y = jnp.array([1, -1])

        gp.fit(X, Y)

        self.assertTrue(jnp.allclose(Y, gp.predict(X)))

    def test_learned_parameters_calcualation(self):
        gp = GaussPerceptron()
        gp._w = jnp.array([1, -1, -1, 4, 1, 1, 1])
        gp._ndim = 2
        gp._is_fitted = True

        K, m, theta = gp.learned_parameters

        expected_K = jnp.array([[4/3, 1/3],
                                [1/3, 1/3]])
        self.assertTrue(jnp.allclose(K, expected_K))

        expected_m = jnp.array([-5/6, -1/3])
        self.assertTrue(jnp.allclose(m, expected_m))

        expected_theta = 0.39
        self.assertTrue(jnp.allclose(theta, expected_theta, atol=10e-1))
