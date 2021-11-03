from abc import ABC, abstractmethod
from typing import Callable

import jax
from jax import numpy as jnp


class PerceptronBase(ABC):

    _w: jnp.ndarray  # for type checking

    def __init__(self,
                 max_niter: int,
                 kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.number] = jax.jit(jnp.dot)) -> None:
        assert max_niter > 0

        super().__init__()
        self._kernel = kernel
        self._is_fitted = False
        self._max_niter = max_niter

    @property
    def w(self) -> jnp.ndarray:
        return self._w

    @property
    def is_fitted(self):
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError('Model does not fitted')

    def _transform_X(self, X: jnp.ndarray) -> jnp.ndarray:
        return X

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        self._check_is_fitted()

        T = self._transform_X(X)
        if T.shape[1] != self._w.shape[0]:
            raise ValueError('Incorrect number of features')

        res = self._kernel(T, self._w) > 0
        return jnp.where(res, 1, -1)

    @abstractmethod
    def fit(self, X: jnp.ndarray, y: jnp.ndarray):  # return self
        pass
