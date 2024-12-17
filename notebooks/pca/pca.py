from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        # Fit the model with X
        return self

    def transform(self, X):
        # Apply the dimensionality reduction on X
        return X

    def fit_transform(self, X, y=None):
        # Fit to data, then transform it
        self.fit(X, y)
        return self.transform(X)