from typing import Optional, Self
from .kernels import Kernel

import numpy as np
import scipy.linalg as la

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class KernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 2, kernel: Optional[Kernel] = None,
                 fit_inverse_transform: bool = False, alpha: float = 1.0) -> None:
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

        self.fit_inverse_transform = fit_inverse_transform

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Self:
        """Fit the Kernel PCA model on the data."""
        K = self.kernel(X, X)

        self.K_ = self._center_kernel(K)

        eigenvalues, eigenvectors = np.linalg.eigh(self.K_)

        self.eigenvalues_ = eigenvalues[::-1]
        self.eigenvectors_ = eigenvectors[:, ::-1]

        self.alphas_ = self.eigenvectors_[:, :self.n_components]
        self.lambdas_ = self.eigenvalues_[:self.n_components]

        self.X_fit_ = X


        if self.fit_inverse_transform:
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
        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)
        K = self.kernel(X_transformed)
        K.flat[::n_samples + 1] += self.alpha
        self.dual_coef_ = la.solve(K, self.X_fit_, assume_a="pos", overwrite_a=True)
        self.X_transformed_fit_ = X_transformed

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Approximate the inverse transformation from feature space to input space."""
        if not self.fit_inverse_transform:
            raise ValueError("Inverse transformation was not enabled during model instantiation.")
        if self.dual_coef_ is None:
            raise NotFittedError("The model must be fitted before performing inverse transformation.")

        K = self.kernel(X_transformed, self.X_transformed_fit_)
        return K @ self.dual_coef_
