import numpy as np
from .metrice import Metric

class Precision(Metric):
    def compute(self, y_pred, target):
        if y_pred.ndim > 1:
            preds = np.argmax(y_pred, axis=1)
        else :
            preds = (y_pred >= 0.5).astype(int)
        
        tp = np.sum((preds == 1) & (target == 1))
        fp = np.sum((preds == 1) & (target == 0))

        if tp+fp == 0:
            return 0
        return tp / (tp + fp)