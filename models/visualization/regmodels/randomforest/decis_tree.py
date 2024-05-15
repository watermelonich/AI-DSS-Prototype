import numpy as np
from collections import Counter
import sys

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
    
    # def _grow_tree(self, x, y, depth = 0):
    #         n_samples, n_feats = x.shape
    #         n_labels = len(np.unique(y))

    #         with open("decis_tree_output.txt", "a") as f:
    #             sys.stdout = f

    #             if (depth>self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
    #                 leaf_value = self._most_common_labels(y)
    #                 print(f"\nReached leaf node at depth {depth}. Predicted value: {leaf_value}\n")
    #                 sys.stdout = sys.__stdout__
    #                 return Node(value=leaf_value)

    #             feat_idxs = np.random.choice(n_feats, self.n_feautures, replace=False)

    #             best_feature, best_thresh = self._best_split(x, y, feat_idxs)
    #             print(f"At depth {depth}, splitting on feature {best_feature} with threshold {best_thresh}.\n")

    #             left_idxs, right_idxs = self._split(x[:, best_feature], best_thresh)

    #             left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
    #             right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth+1)

    #             sys.stdout = sys.__stdout__
    #             return Node(best_feature, best_thresh, left, right)
            
