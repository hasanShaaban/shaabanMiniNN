from nn_lib.layers.activations.relu import ReLU
from nn_lib.layers.activations.sigmoid import Sigmoid
from nn_lib.layers.dense import Dense
from nn_lib.losses.mse import MeanSquaredError
from nn_lib.models.neural_network import NeuralNetwork
from nn_lib.models.sequential import Sequential
from nn_lib.optimizers.sgd import SGD
import numpy as np
from nn_lib.utils.numerical_grad import gradient_check

np.random.seed(0)

model = Sequential([
    Dense(3,5),
    ReLU(),
    Dense(5,2)
])

loss_fn = MeanSquaredError()

x = np.random.randn(5,3)
y = np.random.randn(5,2)

gradient_check(model, loss_fn, x, y)