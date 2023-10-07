import numpy as np

class ElasticNetRegressor:
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha 
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None

    def _soft_thresholding_operator(self, x, lambda_):
        """Soft thresholding operator for L1 regularization."""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.coef_)
            residuals = y - y_pred

            # Update coefficients using gradient descent
            gradient = -2 * np.dot(X.T, residuals) + self.alpha * (self.l1_ratio * np.sign(self.coef_) + (1 - self.l1_ratio) * 2 * self.coef_)
            self.coef_ -= self.learning_rate * gradient

    def predict(self, X):
        return np.dot(X, self.coef_)
