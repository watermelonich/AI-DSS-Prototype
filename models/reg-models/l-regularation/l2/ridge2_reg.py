import numpy as np

class RidgeRegressor:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha  # Regularization strength
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.n_iterations = n_iterations  # Number of iterations for gradient descent
        self.coef_ = None  # Model coefficients

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.coef_)
            residuals = y - y_pred

            # Update coefficients using gradient descent with L2 penalty (Ridge)
            gradient = -2 * np.dot(X.T, residuals) + 2 * self.alpha * self.coef_
            self.coef_ -= self.learning_rate * gradient

    def predict(self, X):
        return np.dot(X, self.coef_)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Load your dataset here or replace this with your data loading code
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the custom Ridge model
    ridge = RidgeRegressor(alpha=0.01)  # Adjust alpha for stronger/weaker regularization
    ridge.fit(X_train, y_train)

    # Predict using the trained model
    y_pred = ridge.predict(X_test)

    # Calculate and print the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (Ridge): {mse}")
