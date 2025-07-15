import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

class DecisionTree:
    def __init__(self, criterion: str = 'information_gain', max_depth: Optional[int] = None):
        assert criterion in {'information_gain', 'majority_error', 'gini_index'}
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data: List[Dict[str, str]], attributes: List[str], label_name: str = 'label'):
        self.tree = self._id3(data, attributes, label_name, depth=0)

    def predict(self, data: List[Dict[str, str]]) -> List[str]:
        return [self._predict_one(self.tree, x) for x in data]

    def _predict_one(self, node, x):
        while isinstance(node, dict):
            attr = node['attr']
            if x[attr] in node['children']:
                node = node['children'][x[attr]]
            else:
                # Unseen attribute value: use majority class at this node
                return node['majority']
        return node

    def _id3(self, data, attributes, label_name, depth):
        labels = [row[label_name] for row in data]
        majority_label = Counter(labels).most_common(1)[0][0]
        # Stopping conditions
        if len(set(labels)) == 1:
            return labels[0]
        if not attributes or (self.max_depth is not None and depth >= self.max_depth):
            return majority_label
        # Choose best attribute
        best_attr = self._choose_attribute(data, attributes, label_name)
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

    def _choose_attribute(self, data, attributes, label_name):
        if self.criterion == 'information_gain':
            scores = [self._information_gain(data, attr, label_name) for attr in attributes]
            return attributes[scores.index(max(scores))]
        elif self.criterion == 'majority_error':
            scores = [self._majority_error_gain(data, attr, label_name) for attr in attributes]
            return attributes[scores.index(max(scores))]
        elif self.criterion == 'gini_index':
            scores = [self._gini_gain(data, attr, label_name) for attr in attributes]
            return attributes[scores.index(max(scores))]

    def _entropy(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return -sum((count/total) * math.log2(count/total) for count in counts.values() if count)

    def _information_gain(self, data, attr, label_name):
        labels = [row[label_name] for row in data]
        base_entropy = self._entropy(labels)
        attr_values = set(row[attr] for row in data)
        weighted_entropy = 0.0
        for value in attr_values:
            subset = [row for row in data if row[attr] == value]
            if not subset:
                continue
            subset_labels = [row[label_name] for row in subset]
            weighted_entropy += (len(subset)/len(data)) * self._entropy(subset_labels)
        return base_entropy - weighted_entropy

    def _majority_error(self, labels):
        total = len(labels)
        if total == 0:
            return 0.0
        majority = Counter(labels).most_common(1)[0][1]
        return 1 - (majority / total)

    def _majority_error_gain(self, data, attr, label_name):
        labels = [row[label_name] for row in data]
        base_me = self._majority_error(labels)
        attr_values = set(row[attr] for row in data)
        weighted_me = 0.0
        for value in attr_values:
            subset = [row for row in data if row[attr] == value]
            if not subset:
                continue
            subset_labels = [row[label_name] for row in subset]
            weighted_me += (len(subset)/len(data)) * self._majority_error(subset_labels)
        return base_me - weighted_me

    def _gini(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return 1 - sum((count/total)**2 for count in counts.values())

    def _gini_gain(self, data, attr, label_name):
        labels = [row[label_name] for row in data]
        base_gini = self._gini(labels)
        attr_values = set(row[attr] for row in data)
        weighted_gini = 0.0
        for value in attr_values:
            subset = [row for row in data if row[attr] == value]
            if not subset:
                continue
            subset_labels = [row[label_name] for row in subset]
            weighted_gini += (len(subset)/len(data)) * self._gini(subset_labels)
        return base_gini - weighted_gini 