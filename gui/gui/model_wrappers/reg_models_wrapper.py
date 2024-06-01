# # import os
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from models.reg_models.decisiontreereg.decis_tree import DecisionTreeRegressor
# # # Import other models as needed

# # def run_decision_tree_regressor():
# #     model = DecisionTreeRegressor()
# #     data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'models', 'reg_models', 'decisiontreereg', 'data_sample.csv'))
# #     model.train(data)
# #     predictions = model.predict(data)
# #     plt.figure()
# #     plt.plot(data['Feature'], data['Target'], 'b-', label='Actual')
# #     plt.plot(data['Feature'], predictions, 'r-', label='Predicted')
# #     plt.legend()
# #     plt.title('Decision Tree Regressor')
# #     plt.xlabel('Feature')
# #     plt.ylabel('Target')
# #     plot_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'plots', 'decision_tree_regressor.png')
# #     plt.savefig(plot_path)
# #     return plot_path

# import sys
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from model_wrappers.reg_models.decisiontreereg.decis_tree import DecisionTreeRegressor

# def run_decision_tree_regressor():
#     model = DecisionTreeRegressor()
#     data_path = os.path.join(os.path.dirname(__file__), 'models', 'reg_models', 'decisiontreereg', 'data_sample.csv')
#     data = pd.read_csv(data_path)
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.plot(data['Feature'], data['Target'], 'b-', label='Actual')
#     plt.plot(data['Feature'], predictions, 'r-', label='Predicted')
#     plt.legend()
#     plt.title('Decision Tree Regressor')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gui', 'plots', 'decision_tree_regressor.png')
#     plt.savefig(plot_path)
#     return plot_path

# Imports
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt

# Utility Functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Base Models

## Linear Regression
class LinearRegression:
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
            y_pred = np.dot(x, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

## Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior += prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

## K-Nearest Neighbors
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        return [self._predict(e) for e in x]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Tree-Based Models

## Decision Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, x, y):
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        n_samples, n_feats = x.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(x, y, feat_idxs)
        left_idxs, right_idxs = self._split(x[:, best_feat], best_thresh)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, x, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)

            for threshold in thresholds:
                gain = self._information_gain(x_column, threshold, y)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, x_column, threshold, y):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(x_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self, x_column, split_thresh):
        left_idxs = np.argwhere(x_column <= split_thresh).flatten()
        right_idxs = np.argwhere(x_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, x):
        return np.array([self._traverse_tree(e, self.root) for e in x])

    def _traverse_tree(self, e, node):
        if node.is_leaf_node():
            return node.value
        if e[node.feature] <= node.threshold:
            return self._traverse_tree(e, node.left)
        return self._traverse_tree(e, node.right)

## Random Forest
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(pred) for pred in tree_preds])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

# Regression Models

## Ridge Regression
class RidgeRegressor:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.coef_)
            residuals = y - y_pred

            gradient = -2 * np.dot(X.T, residuals) + 2 * self.alpha * self.coef_
            self.coef_ -= self.learning_rate * gradient / n_samples

    def predict(self, X):
        return np.dot(X, self.coef_)

# Main Script

def main():
    # Load dataset
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate RandomForest
    rf = RandomForest(n_trees=100, max_depth=10, min_samples_split=2, n_features=None)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True)
    plt.show()

if __name__ == "__main__":
    main()
