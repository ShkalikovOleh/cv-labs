from typing import Tuple
from jax import numpy as jnp

from .PerceptronBase import PerceptronBase


class GaussPerceptron(PerceptronBase):

    def __init__(self, max_niter: int = 10**4) -> None:
        super().__init__(max_niter)

    @property
    def learned_parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.number]:
        self._check_is_fitted()
        n = self._ndim

        Kinv = self._w[:n**2].reshape(n, n)
        K = jnp.linalg.inv(Kinv)

        v = self._w[n**2:n**2+n]
        mu = -0.5 * v @ K

        theta = jnp.exp(0.5 * (self._w[-1] - mu.T @ Kinv @ mu))
        theta = theta / jnp.sqrt((2*jnp.pi)**n * jnp.linalg.det(K))

        return K, mu, theta

    def _transform_X(self, X: jnp.ndarray) -> jnp.ndarray:
        m = X.shape[0]
        n = X.shape[1]            
        X2 = jnp.einsum('...i,...j', X, X).reshape(m, n**2)  # outer prod along last axis
        T = jnp.concatenate([X2, X, jnp.ones((m, 1))], axis=1)
        return T

    def fit(self, X: jnp.ndarray, y: jnp.ndarray):
        n_iter: int = 0
        is_covergence = False

        n = X.shape[1]
        self._ndim = n

        T = self._transform_X(X)
        
        self._w = jnp.zeros(n**2 + n + 1)

        while not is_covergence or n_iter <= self._max_niter:
            n_iter += 1

            # determine where perceptron fail
            fail = y * (T @ self._w) <= 0
            if fail.sum() != 0:
                idx = jnp.argwhere(fail)[0, 0]
                k = y[idx]
                x = T[idx, :]
            else:
                # check that covariance matrix is positive definite
                K = self._w[:n**2].reshape(n, n)
                eig_vals, eig_vecs = jnp.linalg.eigh(K)
                if jnp.any(eig_vals <= 0):
                    # generate correction vector for handling covariance positive-definiteness
                    # single vector, because after correction matrix change
                    eig = eig_vecs[:, jnp.argwhere(eig_vals <= 0)[0, 0]]
                    eig2 = jnp.outer(eig, eig).reshape(n**2)
                    x = jnp.concatenate([eig2, jnp.zeros(n+1)])
                    k = 1
                else:                                        
                    is_covergence = True
                    break                

            self._w = self._w + k*x

        self._is_fitted = n_iter != self._max_niter
        if not self._is_fitted:
            raise ValueError('Fit have finished by iteration limit. \
                            Please check separability of your dataset or max_niter parametr')

        return self
