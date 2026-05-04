# decision_tree.py
import numpy as np
from collections import Counter

class Node:
    """
    A foundational data structure to hold the tree.
    If 'value' is not None, this is a leaf node holding a class prediction.
    Otherwise, it's a decision node holding a feature index and threshold.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        """
        Initializes the Decision Tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param max_depth: Maximum depth of the tree to prevent overfitting.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Builds the tree via recursion."""
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 1. Check stopping criteria (Max depth reached, node is pure, or too few samples)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 2. Find the best split
        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # 3. Create child nodes recursively
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """Iterates through all features and unique values to find the split with the highest Information Gain."""
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """Calculates Information Gain = Entropy(parent) - [Weighted Average] * Entropy(children)"""
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        """Splits the array based on a threshold."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """Calculates Shannon Entropy: -Sum(p * log2(p))"""
        hist = np.bincount(y)
        ps = hist / len(y)
        # Add a tiny epsilon to prevent log(0) errors
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Traverses the tree to make predictions for an array of inputs."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Recursively walks down the tree until it hits a leaf node."""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# --- Quick Test ---
if __name__ == "__main__":
    # Fake dataset: [Age, Tumor Size (mm)]
    # Class 0: Benign, Class 1: Malignant
    X_train = np.array([
        [25, 2.1], [30, 1.8], [28, 2.5],  # Benign
        [65, 8.5], [55, 7.2], [70, 9.1],  # Malignant
        [45, 5.0], [50, 4.5]              # Borderline cases
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1, 0, 1])

    print("🧠 Growing Decision Tree...")
    clf = DecisionTree(max_depth=3)
    clf.fit(X_train, y_train)

    X_test = np.array([[32, 2.0], [60, 8.0], [48, 4.8]])
    predictions = clf.predict(X_test)
    
    print(f"🎯 Predictions for new patients: {predictions}")
    print("Expected roughly: [0, 1, ?]")