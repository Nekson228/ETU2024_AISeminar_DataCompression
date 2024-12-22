from typing import Optional, Self
from kernels import Kernel

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class KernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 2, kernel: Optional[Kernel] = None, alpha: float = 1.0) -> None:
        """
        Parameters:
        - n_components: principal components amount to keep.
        - kernel: An instance of a class implementing the Kernel interface.
        - alpha: Regularization parameter for the inverse transform.
        """
        self.n_components = n_components
        if kernel is None or not isinstance(kernel, Kernel):
            raise ValueError("A valid kernel object implementing the Kernel interface must be provided.")
        self.kernel = kernel
        self.alpha = alpha

        self.K_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: np.ndarray | None = None

        self.alphas_: np.ndarray | None = None
        self.lambdas_: np.ndarray | None = None

        self.X_fit_: np.ndarray | None = None
        self.dual_coef_: np.ndarray | None = None

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

        # Compute dual coefficients for inverse transform
        self._fit_inverse_transform()

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

    def _fit_inverse_transform(self) -> None:
        """Compute dual coefficients for an inverse transform using ridge regression."""
        n_samples = self.X_fit_.shape[0]
        K = self.K_ + self.alpha * np.eye(n_samples)
        self.dual_coef_ = np.linalg.solve(K, self.X_fit_)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Approximate the inverse transformation from feature space to input space."""
        if self.dual_coef_ is None:
            raise ValueError("The model must be fitted before performing inverse transformation.")

        K = self.kernel(X_transformed, self.alphas_)
        return K @ self.dual_coef_
