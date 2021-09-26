from abc import ABC, abstractmethod
from typing import Tuple

from jax import numpy as jnp
from jax import random as rand


class MixtureModelBase(ABC):

    def __init__(self, n_cluster: int, random_state: int = 42, n_iter: int = 10) -> None:
        assert n_cluster > 0
        assert random_state >= 0
        assert n_iter > 0

        super().__init__()

        self._k = n_cluster
        self._K = jnp.arange(n_cluster)
        self._rand_key = rand.PRNGKey(random_state)
        self._n_iter = n_iter
        self.__is_fitted = False

    @abstractmethod
    def _initialize_p_mixture(self) -> jnp.ndarray:
        pass

    @abstractmethod
    def _initialize_dist_params(self, m: int) -> jnp.ndarray:
        pass

    @abstractmethod
    def _E_step(self, dist_params: jnp.ndarray, p_mixture: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def _M_step(self, p_clusters: jnp.ndarray, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass

    def predict(self, X: jnp.ndarray, ret_proba: bool = False) -> jnp.ndarray:
        if not self.__is_fitted:
            raise ValueError('Model does not fitted')
        if X.shape[1] != self.dist_params.shape[1]:
            raise ValueError('Incorrect number of features')

        proba = self._E_step(self.dist_params, self.p_mixture, X)
        if ret_proba:
            return proba
        else:
            return proba.argmax(axis=0)

    def fit_predict(self, X: jnp.ndarray, ret_proba: bool = False) -> jnp.ndarray:
        self.p_mixture = self._initialize_p_mixture()  # init mixture weights
        self.dist_params = self._initialize_dist_params(X.shape[1])

        for i in range(self._n_iter):
            # calculate probabilities of clusters given data
            p_clusters = self._E_step(self.dist_params, self.p_mixture, X)

            # update mixture weights and dist params
            self.dist_params, self.p_mixture = self._M_step(p_clusters, X)

        self.__is_fitted = True

        if ret_proba:
            return p_clusters
        else:
            return p_clusters.argmax(axis=0)
