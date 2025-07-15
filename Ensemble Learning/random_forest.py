import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DecisionTree')))
from id3 import DecisionTree
from collections import Counter

class RandomTree(DecisionTree):
    def __init__(self, k):
        super().__init__(criterion='information_gain', max_depth=None)
        self.k = k

    def _id3(self, data, attributes, label_name, depth):
        labels = [row[label_name] for row in data]
        majority_label = Counter(labels).most_common(1)[0][0]
        if len(set(labels)) == 1:
            return labels[0]
        if not attributes:
            return majority_label
        # Randomly select k features for this split
        if len(attributes) <= self.k:
            candidate_attrs = attributes
        else:
            candidate_attrs = random.sample(attributes, self.k)
        best_attr = self._choose_attribute(data, candidate_attrs, label_name)
        tree = {'attr': best_attr, 'children': {}, 'majority': majority_label}
        attr_values = set(row[best_attr] for row in data)
        for value in attr_values:
            subset = [row for row in data if row[best_attr] == value]
            if not subset:
                tree['children'][value] = majority_label
            else:
                remaining_attrs = [a for a in attributes if a != best_attr]
                tree['children'][value] = self._id3(subset, remaining_attrs, label_name, depth + 1)
        return tree

class RandomForest:
    def __init__(self, T=100, k=2):
        self.T = T
        self.k = k
        self.trees = []

    def fit(self, X, y, features, T=None, k=None):
        if T is not None:
            self.T = T
        if k is not None:
            self.k = k
        n = len(X)
        self.trees = []
        for t in range(self.T):
            idxs = np.random.choice(n, n, replace=True)
            X_boot = [X[i] for i in idxs]
            y_boot = [y[i] for i in idxs]
            data_boot = [dict(x) for x in X_boot]
            for i, label in enumerate(y_boot):
                data_boot[i]['label'] = label
            tree = RandomTree(self.k)
            tree.fit(data_boot, features, label_name='label')
            self.trees.append(tree)

    def predict(self, X):
        n = len(X)
        votes = []
        data = [dict(x) for x in X]
        for tree in self.trees:
            votes.append(tree.predict(data))
        votes = np.array(votes)  # shape: (T, n)
        preds = []
        for i in range(n):
            count = Counter(votes[:, i])
            preds.append(count.most_common(1)[0][0])
        return np.array(preds) 