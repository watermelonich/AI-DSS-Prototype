import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class LassoRegressor:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha
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

            # Update coefficients using gradient descent with L1 penalty
            gradient = -2 * np.dot(X.T, residuals) + self.alpha * self._soft_thresholding_operator(self.coef_, self.alpha)
            self.coef_ -= self.learning_rate * gradient

    def predict(self, X):
        return np.dot(X, self.coef_)

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 5)
true_coefficients = np.array([1, 2, 3, 4, 5])
noise = np.random.normal(0, 1, 100)
y = X.dot(true_coefficients) + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Lasso model
lasso = LassoRegressor(alpha=0.01)  # Adjust alpha for stronger/weaker regularization
lasso.fit(X_train, y_train)

# Predict using the trained model
y_pred = lasso.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Lasso): {mse}")
