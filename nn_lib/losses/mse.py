import numpy as np
from .loss import Loss

class MeanSquaredError(Loss):
    

    def _loss(self):
        diff = self.y_pred - self.target
        loss = 0.5 * np.mean(np.sum(diff**2, axis=1))
        return loss
    
    def _gradient(self):
        N = self.y_pred.shape[0]
        grad = (self.y_pred - self.target) / N
        return grad