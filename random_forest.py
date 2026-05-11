# random_forest.py
import numpy as np
from collections import Counter

# Look at this! Importing our own custom module from Day 4
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10):
        """
        Initializes the Random Forest ensemble.
        :param n_trees: The number of decision trees to plant in our forest.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Trains the forest using Bootstrap Aggregating (Bagging).
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Initialize an untrained tree using our Day 4 class
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            
            # 1. Create a random bootstrap subset of the data
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 2. Train this specific tree only on the random subset
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        """
        Picks random rows from the dataset WITH replacement.
        Some rows will be picked multiple times, some not at all.
        """
        n_samples = X.shape[0]
        # Generate an array of random indices
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """
        Collects predictions from all trees and returns the majority vote per sample.
        """
        # Collect predictions from each tree in the forest
        # Shape of tree_preds: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Swap axes so we can iterate through predictions sample-by-sample
        # New shape: (n_samples, n_trees)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Perform majority voting for each test sample
        return np.array([self._most_common_label(sample_preds) for sample_preds in tree_preds])

    def _most_common_label(self, y):
        """Helper function to find the winning vote."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]


# --- Quick Test ---
if __name__ == "__main__":
    # Let's generate a larger, noisier dataset to watch the ensemble shine
    np.random.seed(42)
    
    print("📊 Generating noisy classification data...")
    # Class 0 centered around (0,0), Class 1 centered around (2,2)
    X_class0 = np.random.randn(50, 2) + np.array([0, 0])
    X_class1 = np.random.randn(50, 2) + np.array([2, 2])
    
    X_train = np.vstack([X_class0, X_class1])
    y_train = np.array([0] * 50 + [1] * 50)
    
    print("🧠 Planting a Random Forest with 5 trees...")
    rf = RandomForest(n_trees=5, max_depth=3)
    rf.fit(X_train, y_train)
    
    # Test on extreme points and one tricky boundary point
    X_test = np.array([
        [-2.0, -2.0],  # Definitely Class 0
        [4.0, 4.0],    # Definitely Class 1
        [1.0, 1.0]     # Right in the messy middle boundary
    ])
    
    predictions = rf.predict(X_test)
    print(f"🎯 Ensemble Predictions: {predictions}")
    print("Expected roughly: [0, 1, ? depending on the forest's vote]")