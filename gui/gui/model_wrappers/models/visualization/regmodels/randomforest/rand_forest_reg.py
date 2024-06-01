import numpy as np
from collections import Counter

class Node:
    def __init__(self, feauture=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feauture
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth = 100, n_feautures = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feautures = n_feautures
        self.root = None

    def fit(self, x, y):
        self.n_feautures = x.shape[1] if not self.n_feautures else min(x.shape[1], self.n_feautures)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth = 0):
        n_samples, n_feats = x.shape
        n_labels = len(np.unique(y))

        if (depth>self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
            leaf_value = self._most_common_labels(y)
            return Node (value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_feautures, replace=False)

        best_feature, best_thresh = self._best_split(x, y, feat_idxs)

        left_idxs, right_idxs = self._split(x[:, best_feature], best_thresh)

        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs, :], y [right_idxs], depth+1)

        return Node(best_feature, best_thresh, left, right)
        
    def _best_split(self, x, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)

            for thr in thresholds:
                gain = self._information_gain(x_column, thr, y)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def _information_gain(self, x_column, threshold, y):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(x_column, threshold)

        if len (left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len (y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y [left_idxs]), self._entropy(y [right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, x_column, split_thresh):
        left_idxs = np.argwhere(x_column <= split_thresh).flatten()
        right_idxs = np.argwhere(x_column > split_thresh).flatten()

        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)

        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_labels(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, x):
        return np.array([self._traverse_tree(e, self.root) for e in x])
    
    def _traverse_tree(self, e, node):
        if node.is_leaf_node():
            return node.value
        
        if e [node.feature] <= node.threshold:
            return self._traverse_tree(e, node.left) 
        return self._traverse_tree(e, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth, min_samples_split = self.min_samples_split, n_features = self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    