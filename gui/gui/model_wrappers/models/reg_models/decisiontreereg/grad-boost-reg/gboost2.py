import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def _calculate_residuals(self, y, y_pred):
        return y - y_pred

    def _fit_tree(self, X, y, residuals):
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X, residuals)
        return tree

    def fit(self, X, y):
        y_pred = np.zeros_like(y)

        for _ in range(self.n_estimators):
            residuals = self._calculate_residuals(y, y_pred)
            tree = self._fit_tree(X, y, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

