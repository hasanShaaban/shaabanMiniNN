from nn_lib.layers.activations.relu import ReLU
from nn_lib.layers.dense import Dense
from nn_lib.losses.sofmax_cross_entropy import SoftMaxCrossEntropy
from nn_lib.models.neural_network import NeuralNetwork
import numpy as np

net = NeuralNetwork()

net.add(Dense(2,5))
net.add(ReLU())
net.add(Dense(5,3))

net.add(ReLU())


x = np.random.randn(5,2)
out = net.forward(x)

func = SoftMaxCrossEntropy()
target = np.array([[2], [1], [1], [3], [3]])
loss = func.forward(out, target)
params = net.get_params()

print(out)
print(loss)
print(params)