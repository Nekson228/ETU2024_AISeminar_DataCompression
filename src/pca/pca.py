from typing import Self

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

import numpy as np


class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

        self._components_: np.ndarray | None = None
        self._explained_variance_: np.ndarray | None = None
        self._overall_variance_: float | None = None

        self._mean_: np.ndarray | None = None
        self._std_: np.ndarray | None = None

    @property
    def components_(self) -> np.ndarray:
        return self._components_

    @property
    def explained_variance_(self) -> np.ndarray:
        return self._explained_variance_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._explained_variance_ / self._overall_variance_

    @property
    def mean_(self) -> np.ndarray:
        return self._mean_

    @property
    def std_(self) -> np.ndarray:
        return self._std_

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        check_array(X)

        self._mean_, self._std_ = np.mean(X, axis=0), np.std(X, axis=0)
        self._std_[self._std_ == 0] = 1
        X = (X - self.mean_) / self.std_

        cov = np.cov(X.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)  # eigh is used for symmetric (hermitian) matrices

        self._overall_variance_ = np.sum(eig_vals)

        idx = np.argsort(eig_vals)[::-1]
        self._components_ = eig_vecs[:, idx]
        self._explained_variance_ = eig_vals[idx]

        if self.n_components is not None:
            self._components_ = self.components_[:, :self.n_components]
            self._explained_variance_ = self.explained_variance_[:self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_array(X)

        X = (X - self.mean_) / self.std_
        X = X @ self.components_

        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, **fit_params) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_array(X)

        return X @ self.components_.T * self.std_ + self.mean_
