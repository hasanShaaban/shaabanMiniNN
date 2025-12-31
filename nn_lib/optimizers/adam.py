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
            param = params[key]
            grad = grads[key]

            pid = id(param)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(param)
                self.v[pid] = np.zeros_like(param)

            self.m[pid] = (self.beta1 * self.m[pid] + (1-self.beta1) * grad)
            self.v[pid] = (self.beta2 * self.v[pid] + (1-self.beta2) * (grad**2))

            m_hat = self.m[pid] / (1-self.beta1 ** self.t)
            v_hat = self.v[pid] / (1-self.beta2 ** self.t)

            param -= (self.lr * m_hat/ (np.sqrt(v_hat) + self.eps))