from typing import Tuple

from jax import jit
from jax import numpy as jnp
from jax import random as rand
from jax import vmap

from .MixtureModelBase import MixtureModelBase


class BMM(MixtureModelBase):

    def __init__(self, n_cluster: int, random_state: int = 42,
                 n_iter: int = 10, eps: float = 10**-2) -> None:
        assert 1 > eps >= 0

        super().__init__(n_cluster, random_state=random_state, n_iter=n_iter)
        self.__eps = eps
        self.__E_impl = self.__E_step_init()
        self.__M_impl = self.__M_step_init()

    def _initialize_p_mixture(self) -> jnp.ndarray:
        p = rand.uniform(self._rand_key, shape=(self._k, ))
        return p / p.sum()  # norm probabilities

    def _initialize_dist_params(self, m: int) -> jnp.ndarray:
        return rand.uniform(self._rand_key, shape=(self._k, m))

    def __E_step_init(self):
        def E(dist_params, p_mixture, k, x, eps):
            sum = 0
            # clip for avoiding zero(and near zero) division
            dist_params = jnp.clip(dist_params, eps, 1-eps)

            # sum along all K, because jax vmap does not support if
            for g in range(p_mixture.shape[0]):
                t1 = (dist_params[g] / dist_params[k])**x
                t2 = ((1-dist_params[g]) / (1-dist_params[k]))**(1-x)
                t3 = p_mixture[g] / p_mixture[k]
                sum += t3 * jnp.prod(t1 * t2)

            return 1 / sum

        E = vmap(E, (None, None, None, 0, None))  # map over data points
        return jit(vmap(E, (None, None, 0, None, None)))  # map over number of clusters and jit

    def __M_step_init(self):
        def M(p_clusters, X):
            sum = p_clusters.sum(axis=1)  # for reusing below
            p_mixture = sum / X.shape[0]
            dist_params = (p_clusters @ X) / jnp.expand_dims(sum, 1) # sum over axis is matvec multiplication!!!
            return dist_params, p_mixture

        return jit(M)

    def _E_step(self, dist_params: jnp.ndarray, p_mixture: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        return self.__E_impl(dist_params, p_mixture, self._K, X, self.__eps)

    def _M_step(self, p_clusters: jnp.ndarray, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if (p_clusters == jnp.inf).any() or jnp.isnan(p_clusters).any():
            raise ValueError('Internal BMM error. Try increase eps parameter')
        return self.__M_impl(p_clusters, X)
