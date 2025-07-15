import numpy as np
from decision_stumps import DecisionStump

class AdaBoost:
    def __init__(self, T=50):
        self.T = T
        self.stumps = []
        self.alphas = []

    def fit(self, X, y, features, T=None):
        if T is not None:
            self.T = T
        n = len(X)
        D = np.ones(n) / n
        self.stumps = []
        self.alphas = []
        y_ = np.array(y)
        # Prepare data as list of dicts with label_name='label'
        data = [dict(x) for x in X]
        for i, label in enumerate(y):
            data[i]['label'] = label
        for t in range(self.T):
            stump = DecisionStump()
            stump.fit(data, features, label_name='label', sample_weight=D)
            pred = np.array(stump.predict(data))
            # Convert predictions to {-1, +1}
            pred_num = np.where(pred == 1, 1, -1) if pred.dtype != object else np.array([1 if p == 1 or p == 'yes' else -1 for p in pred])
            err = np.sum(D * (pred_num != y_)) / np.sum(D)
            if err > 0.5:
                break
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            self.stumps.append(stump)
            self.alphas.append(alpha)
            # Update weights
            D = D * np.exp(-alpha * y_ * pred_num)
            D = D / np.sum(D)

    def predict(self, X):
        n = len(X)
        agg = np.zeros(n)
        data = [dict(x) for x in X]
        for alpha, stump in zip(self.alphas, self.stumps):
            pred = np.array(stump.predict(data))
            pred_num = np.where(pred == 1, 1, -1) if pred.dtype != object else np.array([1 if p == 1 or p == 'yes' else -1 for p in pred])
            agg += alpha * pred_num
        return np.where(agg >= 0, 1, -1) 