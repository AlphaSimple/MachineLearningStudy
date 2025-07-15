import numpy as np
from collections import Counter, defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DecisionTree')))
from id3 import DecisionTree

class DecisionStump(DecisionTree):
    def __init__(self):
        super().__init__(criterion='information_gain', max_depth=1)
        self.sample_weight = None

    def fit(self, data, attributes, label_name='label', sample_weight=None):
        self.sample_weight = sample_weight
        self.tree = self._stump(data, attributes, label_name)

    def _stump(self, data, attributes, label_name):
        labels = [row[label_name] for row in data]
        if self.sample_weight is None:
            weights = np.ones(len(labels)) / len(labels)
        else:
            weights = np.array(self.sample_weight)
        # Choose best attribute using weighted information gain
        best_attr = self._choose_attribute_weighted(data, attributes, label_name, weights)
        stump = {'attr': best_attr, 'children': {}, 'majority': self._weighted_majority(labels, weights)}
        attr_values = set(row[best_attr] for row in data)
        for value in attr_values:
            idxs = [i for i, row in enumerate(data) if row[best_attr] == value]
            if not idxs:
                stump['children'][value] = stump['majority']
            else:
                sub_labels = [labels[i] for i in idxs]
                sub_weights = weights[idxs]
                stump['children'][value] = self._weighted_majority(sub_labels, sub_weights)
        return stump

    def _choose_attribute_weighted(self, data, attributes, label_name, weights):
        scores = [self._information_gain_weighted(data, attr, label_name, weights) for attr in attributes]
        return attributes[np.argmax(scores)]

    def _information_gain_weighted(self, data, attr, label_name, weights):
        labels = [row[label_name] for row in data]
        base_entropy = self._entropy_weighted(labels, weights)
        attr_values = set(row[attr] for row in data)
        weighted_entropy = 0.0
        for value in attr_values:
            idxs = [i for i, row in enumerate(data) if row[attr] == value]
            if not idxs:
                continue
            sub_labels = [labels[i] for i in idxs]
            sub_weights = weights[idxs]
            weighted_entropy += sub_weights.sum() / weights.sum() * self._entropy_weighted(sub_labels, sub_weights)
        return base_entropy - weighted_entropy

    def _entropy_weighted(self, labels, weights):
        total = np.sum(weights)
        counter = defaultdict(float)
        for l, w in zip(labels, weights):
            counter[l] += w
        return -sum((w/total) * np.log2(w/total) for w in counter.values() if w > 0)

    def _weighted_majority(self, labels, weights):
        counter = defaultdict(float)
        for l, w in zip(labels, weights):
            counter[l] += w
        return max(counter.items(), key=lambda x: x[1])[0]

    def predict(self, X):
        preds = []
        for row in X:
            v = row.get(self.tree['attr'], None)
            if v in self.tree['children']:
                preds.append(self.tree['children'][v])
            else:
                preds.append(self.tree['majority'])
        return preds 