from typing import Optional, Self
from kernels import Kernel

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class KernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 2, kernel: Optional[Kernel] = None) -> None:
        """
        Parameters:
        - n_components: principal components amount to keep.
        - kernel: An instance of a class implementing the Kernel interface.
        """
        self.n_components = n_components
        if kernel is None or not isinstance(kernel, Kernel):
            raise ValueError("A valid kernel object implementing the Kernel interface must be provided.")
        self.kernel = kernel

        self.K_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: np.ndarray | None = None

        self.alphas_: np.ndarray | None = None
        self.lambdas_: np.ndarray | None = None

        self.X_fit_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Self:
        """Fit the Kernel PCA model on the data."""
        K = self.kernel(X, X)

        self.K_ = self._center_kernel(K)

        eigenvalues, eigenvectors = np.linalg.eigh(self.K_)

        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]

        self.alphas_ = self.eigenvectors_[:, :self.n_components]
        self.lambdas_ = self.eigenvalues_[:self.n_components]

        self.X_fit_ = X

        return self

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        multiplier = np.eye(N) - one_n
        return multiplier @ K @ multiplier

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data into the kernel PCA feature space."""
        K_centered = self._center_kernel(self.kernel(X, self.X_fit_))

        return K_centered @ self.alphas_

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        """Fit the model and transform the input data."""
        self.fit(X, y)
        return self.transform(X)
