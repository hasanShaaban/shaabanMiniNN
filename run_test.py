from nn_lib.layers.activations.relu import ReLU
from nn_lib.layers.dense import Dense
from nn_lib.losses.sofmax_cross_entropy import SoftMaxCrossEntropy
import numpy as np

logits = np.array([[0.0004,2,0.2]])
y = np.array([1])

func = SoftMaxCrossEntropy()
loss = func.forward(logits, y)
grad = func.backward()

print(loss)
print(grad)