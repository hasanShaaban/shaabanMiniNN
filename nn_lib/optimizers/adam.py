import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m ={}
        self.v ={}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = (self.beta1 * self.m[key] + (1-self.beta1) * grads[key])
            self.v[key] = (self.beta2 * self.v[key] + (1-self.beta2) * (grads[key]**2))

            m_hat = self.m[key] / (1-self.beta1 ** self.t)
            v_hat = self.v[key] / (1-self.beta2 ** self.t)

            params[key] -= (self.lr * m_hat/ (np.sqrt(v_hat) + self.eps))