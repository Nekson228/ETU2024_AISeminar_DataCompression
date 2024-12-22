from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    """Abstract base class for all kernel functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
    """
    Linear kernel implementation.

    The linear kernel is defined as:
    K(x, y) = x^T y
    """

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.dot(X, Y.T)


class PolynomialKernel(Kernel):
    """
    Polynomial kernel implementation.

    The polynomial kernel is defined as:
    K(x, y) = (gamma * x^T y + coef0)^degree
    """

    def __init__(self, gamma: float = 1., degree: int = 3, coef0: int = 1) -> None:
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (self.gamma * X @ Y.T + self.coef0) ** self.degree


class RBFKernel(Kernel):
    """
    RBF (Gaussian) kernel implementation.

    The RBF kernel is defined as:
    K(x, y) = exp(-gamma * ||x - y||^2)
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        sq_dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * sq_dists)


class SigmoidKernel(Kernel):
    """Sigmoid kernel implementation."""

    def __init__(self, gamma: float = 1.0, coef0: int = 1) -> None:
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.tanh(np.dot(X, Y.T) * self.gamma + self.coef0)
