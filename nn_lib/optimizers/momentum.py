import numpy as np
from .optimizer import Optimizer

class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            self.v[key] = (
                self.momentum * self.v[key] - self.lr * grads[key]
            )

            params[key] += self.v[key]