import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, lr=0.01, eps = 12-8):
        super().__init__(lr)
        self.eps = eps
        self.cache = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])

            self.cache[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.cache[key]) + self.eps)