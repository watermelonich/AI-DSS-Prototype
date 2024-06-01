# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from models.clas_models.logistic_reg.logist_reg import LogisticRegression
# # Import other models as needed

# def run_logistic_regression():
#     model = LogisticRegression()
#     data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'models', 'clas_models', 'logistic_reg', 'data_sample.csv'))
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.scatter(data['Feature'], data['Target'], label='Actual')
#     plt.scatter(data['Feature'], predictions, label='Predicted')
#     plt.legend()
#     plt.title('Logistic Regression')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'plots', 'logistic_regression.png')
#     plt.savefig(plot_path)
#     return plot_path

# import sys
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Add the parent directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# from models.clas_models.logistic_reg.logist_reg import LogisticRegression
# # Import other models as needed

# def run_logistic_regression():
#     model = LogisticRegression()
#     data_path = os.path.join(os.path.dirname(__file__), 'models', 'clas_models', 'logistic_reg', 'data_sample.csv')
#     data = pd.read_csv(data_path)
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.scatter(data['Feature'], data['Target'], label='Actual')
#     plt.scatter(data['Feature'], predictions, label='Predicted')
#     plt.legend()
#     plt.title('Logistic Regression')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'plots', 'logistic_regression.png')
#     plt.savefig(plot_path)
#     return plot_path

# Imports
import numpy as np

# Utility Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Base Models

## Logistic Regression
class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(x, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(x.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        linear_pred = np.dot(x, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]

        return class_pred

## Principal Component Analysis (PCA)
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X, rowvar=False)

        # Eigenvectors and eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # Transpose eigenvectors for easier calculations
        eigenvectors = eigenvectors.T

        # Sort eigenvectors by eigenvalues in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # Project data onto principal components
        X = X - self.mean
        return np.dot(X, self.components.T)

## Multi-class Logistic Regression
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
                print(f'Iteration {i}, Cost: {cost}')

    def _one_hot_encode(self, y, num_classes):
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        one_hot[np.arange(m), y] = 1
        return one_hot

    def predict(self, X):
        z = np.dot(X, self.theta)
        y_pred = self._softmax(z)
        return np.argmax(y_pred, axis=1)

## Support Vector Machine (SVM)
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
