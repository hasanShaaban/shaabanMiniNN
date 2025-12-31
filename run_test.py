from nn_lib.layers.activations.relu import ReLU
from nn_lib.layers.activations.sigmoid import Sigmoid
from nn_lib.layers.dense import Dense
from nn_lib.losses.mse import MeanSquaredError
from nn_lib.models.neural_network import NeuralNetwork
from nn_lib.models.sequential import Sequential
from nn_lib.optimizers.sgd import SGD
from nn_lib.optimizers.adam import Adam
import numpy as np
from nn_lib.utils.numerical_grad import gradient_check
from nn_lib.training.trainer import Trainer
from nn_lib.layers.batch_norm import BatchNormalization

X = np.array([
    [80,  2, 1, 20, 0],
    [100, 3, 1, 15, 1],
    [120, 3, 2, 10, 1],
    [60,  2, 1, 30, 0],
    [200, 4, 2, 5,  1],
    [150, 4, 2, 8,  0],
    [90,  2, 1, 12, 0],
    [110, 3, 1, 20, 1],
    [130, 3, 2, 6,  1],
    [70,  2, 1, 25, 0],
    [160, 4, 2, 10, 1],
    [140, 3, 2, 15, 0],
    [85,  2, 1, 18, 0],
    [175, 4, 2, 7,  1],
    [95,  3, 1, 10, 0],
], dtype=float)
y = np.array([
    [80*0.5 + 2*20 + 1*15 - 20*1.5 + 0*25 + 10],   # 75.0
    [100*0.5 + 3*20 + 1*15 - 15*1.5 + 1*25 + 10],  # 132.5
    [120*0.5 + 3*20 + 2*15 - 10*1.5 + 1*25 + 10],  # 170.0
    [60*0.5 + 2*20 + 1*15 - 30*1.5 + 0*25 + 10],   # 50.0
    [200*0.5 + 4*20 + 2*15 - 5*1.5  + 1*25 + 10],  # 237.5
    [150*0.5 + 4*20 + 2*15 - 8*1.5  + 0*25 + 10],  # 183.0
    [90*0.5  + 2*20 + 1*15 - 12*1.5 + 0*25 + 10],  # 92.0
    [110*0.5 + 3*20 + 1*15 - 20*1.5 + 1*25 + 10],  # 125.0
    [130*0.5 + 3*20 + 2*15 - 6*1.5  + 1*25 + 10],  # 180.0
    [70*0.5  + 2*20 + 1*15 - 25*1.5 + 0*25 + 10],  # 57.5
    [160*0.5 + 4*20 + 2*15 - 10*1.5 + 1*25 + 10],  # 210.0
    [140*0.5 + 3*20 + 2*15 - 15*1.5 + 0*25 + 10],  # 162.5
    [85*0.5  + 2*20 + 1*15 - 18*1.5 + 0*25 + 10],  # 80.5
    [175*0.5 + 4*20 + 2*15 - 7*1.5  + 1*25 + 10],  # 225.5
    [95*0.5  + 3*20 + 1*15 - 10*1.5 + 0*25 + 10],  # 117.5
], dtype=float)

X_train, y_train = X[:7], y[:7]
X_val, y_val = X[7:11], y[7:11]
X_test, y_test = X[11:], y[11:]

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0) + 1e-8

X_train = (X_train - X_mean) / X_std
X_val   = (X_val - X_mean) / X_std
X_test  = (X_test - X_mean) / X_std



model = Sequential([
    Dense(5, 1),
    
])

loss_fn = MeanSquaredError()
optimizer = SGD(lr=0.01)

trainer = Trainer(
    model = model ,
    loss_fn = loss_fn,
    optimizer= optimizer,
    batch_size = 1,
    epochs=1000,
    shuffle=True
)

trainer.fit(X_train, y_train)

model.summary()

def evaluate(model, loss_fn, X, y):
    model.eval()
    preds = model.forward(X)
    return loss_fn.forward(preds, y)


train_loss = evaluate(model, loss_fn, X_train, y_train)
val_loss   = evaluate(model, loss_fn, X_val, y_val)
test_loss  = evaluate(model, loss_fn, X_test, y_test)

print(f"Train MSE: {train_loss:.4f}")
print(f"Val   MSE: {val_loss:.4f}")
print(f"Test  MSE: {test_loss:.4f}")

def mean_absolute_error(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

model.eval()
test_preds = model.forward(X_test)

W_scaled = model.get_params()[0][1]["W"]
b_scaled = model.get_params()[0][1]["b"]

W_original = W_scaled / X_std.reshape(-1, 1)
b_original = b_scaled - np.sum(W_original.flatten() * X_mean)

print("W:", W_original)
print("b:", b_original)

print("Test MAE:", mean_absolute_error(test_preds, y_test))

