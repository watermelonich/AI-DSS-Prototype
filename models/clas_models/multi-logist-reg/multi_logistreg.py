import numpy as np

class MultiLogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-10)

    def _compute_cost(self, y, y_pred):
        m = y.shape[0]
        l2_reg = (1 / (2 * m)) * np.sum(self.theta**2)
        return (-np.sum(y * np.log(y_pred)) + l2_reg) / m

    def fit(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))
        self.theta = np.zeros((n, num_classes))

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            y_pred = self._softmax(z)
            cost = self._compute_cost(self._one_hot_encode(y, num_classes), y_pred)

            gradient = np.dot(X.T, (y_pred - self._one_hot_encode(y, num_classes))) / m
            self.theta -= self.learning_rate * gradient

            if i % 100 == 0:
                print(f'Iteration {i}')

    def _one_hot_encode(self, y, num_classes):
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y] = 1
        return one_hot

    def predict(self, X):
        z = np.dot(X, self.theta)
        y_pred = self._softmax(z)
        return np.argmax(y_pred, axis=1)
    