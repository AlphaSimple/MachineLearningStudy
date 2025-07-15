import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DecisionTree')))
from id3 import DecisionTree
from collections import Counter

class BaggedTrees:
    def __init__(self, T=100):
        self.T = T
        self.trees = []

    def fit(self, X, y, features, T=None):
        if T is not None:
            self.T = T
        n = len(X)
        self.trees = []
        for t in range(self.T):
            idxs = np.random.choice(n, n, replace=True)
            X_boot = [X[i] for i in idxs]
            y_boot = [y[i] for i in idxs]
            data_boot = [dict(x) for x in X_boot]
            for i, label in enumerate(y_boot):
                data_boot[i]['label'] = label
            tree = DecisionTree(criterion='information_gain', max_depth=None)
            tree.fit(data_boot, features, label_name='label')
            self.trees.append(tree)

    def predict(self, X):
        n = len(X)
        votes = []
        data = [dict(x) for x in X]
        for tree in self.trees:
            votes.append(tree.predict(data))
        votes = np.array(votes)  # shape: (T, n)
        # Majority vote
        preds = []
        for i in range(n):
            count = Counter(votes[:, i])
            preds.append(count.most_common(1)[0][0])
        return np.array(preds) 