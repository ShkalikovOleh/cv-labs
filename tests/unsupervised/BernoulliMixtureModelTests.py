import unittest

from jax import numpy as jnp
from jax import random as rand

from cv.unsupervised import BMM


class BMMTests(unittest.TestCase):

    def test_k_should_be_greater_0(self):
        with self.assertRaises(AssertionError):
            BMM(-1)
        with self.assertRaises(AssertionError):
            BMM(0)

    def test_random_state_should_be_non_negative(self):
        with self.assertRaises(AssertionError):
            BMM(1, random_state=-1)

    def test_n_iteration_should_be_greater_0(self):
        with self.assertRaises(AssertionError):
            BMM(1, n_iter=0)
        with self.assertRaises(AssertionError):
            BMM(1, n_iter=-1)

    def test_eps_should_be_greater_0(self):
        with self.assertRaises(AssertionError):
            BMM(1, eps=-0.1)

    def test_eps_should_be_less_1(self):
        with self.assertRaises(AssertionError):
            BMM(1, eps=1.1)

    def test_E_step(self):
        X = jnp.array([[1], [0], [1]])
        dist_params = jnp.array([[0.7], [0.2]])
        p_mixture = jnp.array([0.66, 1-0.66])

        bmm = BMM(2)        
        p_clusters = bmm._E_step(dist_params, p_mixture, X) # protected method, but this is Python:)))

        t1 = 0.66 * 0.7
        t2 = 0.66 * 0.3
        t3 = (1-0.66) * 0.2
        t4 = (1-0.66) * 0.8
        p_clusters_expected = jnp.array([[t1 / (t1 + t3), t2 / (t2 + t4), t1 / (t1 + t3)],
                                         [t3 / (t1 + t3), t4 / (t2 + t4), t3 / (t1 + t3)]])

        self.assertTrue(jnp.allclose(p_clusters_expected, p_clusters))

    def test_E_step_eps_cliping(self):
        X = jnp.array([[1], [0], [1]])        
        dist_params = jnp.array([[0.96], [0.01]]) # dist params should be cliping to [0.8, 0.2]
        p_mixture = jnp.array([0.66, 1-0.66])

        bmm = BMM(2, eps=0.2)
        p_clusters = bmm._E_step(dist_params, p_mixture, X)

        t1 = 0.66 * 0.8
        t2 = 0.66 * 0.2
        t3 = (1-0.66) * 0.2
        t4 = (1-0.66) * 0.8
        p_clusters_expected = jnp.array([[t1 / (t1 + t3), t2 / (t2 + t4), t1 / (t1 + t3)],
                                         [t3 / (t1 + t3), t4 / (t2 + t4), t3 / (t1 + t3)]])

        self.assertTrue(jnp.allclose(p_clusters_expected, p_clusters))

    def test_M_step(self):
        X = jnp.array([[1], [0], [0]])
        p_clusters = jnp.array([[0.1, 0.5, 0.6],
                                [0.9, 0.5, 0.4]])

        bmm = BMM(2)
        dist_params, p_mixture = bmm._M_step(p_clusters, X)

        p_mixture_expected = jnp.array([0.4, 0.6])
        dist_params_expected = jnp.array([[1/12], [0.5]])

        self.assertTrue(jnp.allclose(p_mixture_expected, p_mixture))
        self.assertTrue(jnp.allclose(dist_params_expected, dist_params))

    def test_predict_dummy(self):
        X = jnp.array([[1, 0],
                       [0, 1]])

        bmm = BMM(2)
        X_pred = bmm.fit_predict(X)
        cluster1 = X_pred[0]

        pred = bmm.predict(jnp.array([[1, 0], [0, 1]]))
        self.assertTrue(pred[0] == cluster1)
        self.assertTrue(pred[1] != cluster1)
