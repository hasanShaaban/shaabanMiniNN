import numpy as np

class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size=32, epochs=10, shuffle=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        
        self.loss_history = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X = X[indices]
                y = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                scores = self.model.forward(X_batch)

                loss = self.loss_fn.forward(scores, y_batch)
                epoch_loss += loss

                dout = self.loss_fn.backward()

                self.model.backward(dout)

                for params, grads in self.model.params_and_grads():
                    self.optimizer.update(params, grads)
            epoch_loss /= (n_samples // self.batch_size)
            self.loss_history.append(epoch_loss)

            print(
                f"Epoch {epoch+1}/{self.epochs}"
                f"Loss: {epoch_loss:.4f}"
            )