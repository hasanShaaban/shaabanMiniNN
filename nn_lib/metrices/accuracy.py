import numpy as np
from .metrice import Metric

class Accuracy(Metric):
    def compute(self, y_pred, target):
        preds = np.argmax(y_pred, axis=1)
        return np.mean(preds == target)