from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    """Abstract base class for all kernel functions."""

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the kernel matrix between X and Y.

        Parameters:
        - X: First input matrix of shape (n_samples, n_features).
        - Y: Optional second input matrix of shape (m_samples, n_features). If None, Y = X.

        Returns:
        - Kernel matrix of shape (n_samples, m_samples).
        """
        if Y is None:
            Y = X
        return self._compute_kernel(X, Y)

    @abstractmethod
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix between X and Y.

        Parameters:
        - X: First input matrix of shape (n_samples, n_features).
        - Y: Second input matrix of shape (m_samples, n_features).

        Returns:
        - Kernel matrix of shape (n_samples, m_samples).
        """
        pass


class LinearKernel(Kernel):
    """Linear kernel implementation."""

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.dot(X, Y.T)


class PolynomialKernel(Kernel):
    """Polynomial kernel implementation."""

    def __init__(self, gamma: float = 1., degree: int = 3, coef0: int = 1) -> None:
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (self.gamma * X @ Y.T + self.coef0) ** self.degree


class RBFKernel(Kernel):
    """RBF (Gaussian) kernel implementation."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        sq_dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * sq_dists)


class SigmoidKernel(Kernel):
    """Sigmoid kernel implementation."""

    def __init__(self, gamma: float = 1.0, coef0: int = 1) -> None:
        self.gamma = gamma
        self.coef0 = coef0

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.tanh(np.dot(X, Y.T) * self.gamma + self.coef0)
