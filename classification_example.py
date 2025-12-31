import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from nn_lib.models.sequential import Sequential
from nn_lib.layers.dense import Dense
from nn_lib.layers.activations.relu import ReLU
from nn_lib.losses.sofmax_cross_entropy import SoftMaxCrossEntropy
from nn_lib.optimizers.adam import Adam
from nn_lib.training.trainer import Trainer
from nn_lib.metrices.accuracy import Accuracy



def load_minst():
    X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True,parser='liac-arff')
    X = X.astype(np.float32)
    y = y.astype(int)

    return X, y

X, y = load_minst()
X /= 255.0

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

y_oh = one_hot(y)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_oh, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

def build_model(hidden_size = 128):
    return Sequential([
        Dense(784, hidden_size),
        ReLU(),
        Dense(hidden_size, hidden_size),
        ReLU(),
        Dense(hidden_size, 10),
    ])

loss_fn = SoftMaxCrossEntropy()
optimizer = Adam(lr=1e-3)

trainer = Trainer(
    model=build_model(),
    loss_fn=loss_fn,
    optimizer=optimizer,
    batch_size=128,
    epochs=100,
    shuffle=True
)

trainer.fit(X_train, y_train)

def evaluate_accuracy(model, X, y_true):
    model.eval()
    scores = model.forward(X)
    preds = np.argmax(scores, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(preds == labels)

train_acc = evaluate_accuracy(trainer.model, X_train, y_train)
val_acc   = evaluate_accuracy(trainer.model, X_val, y_val)
test_acc  = evaluate_accuracy(trainer.model, X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val   Accuracy: {val_acc:.4f}")
print(f"Test  Accuracy: {test_acc:.4f}")
