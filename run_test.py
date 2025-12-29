from nn_lib.layers.activations.relu import ReLU
from nn_lib.layers.dense import Dense
import numpy as np


x = np.random.randn(10, 3)
dense = Dense(3, 5)
relu = ReLU()
out = dense.forward(x)
act = relu.forward(out)
print(act)
