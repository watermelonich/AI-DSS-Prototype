import numpy as np

class SVR:
    def __init__(self, epsilon=0.1, C=1, kernel='rbf', gamma='scale'):
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def _kernel_function(self, X1, X2):
        if self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
        # Add more kernel options as needed

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y

        # Initialize Lagrange multipliers
        alpha = np.zeros(n_samples)
        alpha_prev = np.copy(alpha)

        # Kernel matrix
        K = self._kernel_function(X, X)

        # Training loop
        for _ in range(1000):  # Adjust the number of iterations as needed
            for i in range(n_samples):
                # Calculate predicted output
                y_pred_i = np.dot(alpha * y, K[i]) - alpha[i] * K[i, i]

                # Update Lagrange multiplier
                alpha[i] += self.epsilon * (1 - y_pred_i)

                # Clip alpha values to be within [0, C]
                alpha[i] = max(0, min(alpha[i], self.C))

            # Check for convergence
            if np.linalg.norm(alpha - alpha_prev) < 1e-5:
                break
            alpha_prev = np.copy(alpha)

        # Calculate bias
        support_indices = np.where(alpha > 0)[0]
        self.bias = np.mean(y[support_indices] - np.dot(alpha * y, K[:, support_indices]))

        # Store support vectors and their corresponding labels
        self.support_vectors = X[support_indices]
        self.support_labels = y[support_indices]

    def predict(self, X):
        K = self._kernel_function(X, self.X)
        return np.dot((self.alpha * self.y), K.T) - self.bias
