import numpy as np
from .loss import Loss

class SoftMaxCrossEntropy(Loss):

    def __init__(self):
        super().__init__()
        self.probs = None

    def _loss(self):
        
        y_pred = self.y_pred
        target = self.target
        N = y_pred.shape[0]

        norm_y = y_pred - np.max(y_pred, axis=1, keepdims= True)
        exp_y = np.exp(norm_y)
        self.probs = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        if target.ndim == 1:
            y_onehot = np.zeros_like(self.probs)
            y_onehot[np.arange(N), target] = 1
        else:
            y_onehot = target
        
        loss = np.mean(- np.log(np.sum(self.probs * y_onehot, axis = 1)))
        return loss
    
    def _gradient(self):
        
        N = self.y_pred.shape[0]
        target = self.target

        if target.ndim == 1:
            y_onehot = np.zeros_like(self.probs)
            y_onehot[np.arange(N), target] = 1
        else:
            y_onehot = target

        grad = (self.probs - y_onehot) / N
        return grad