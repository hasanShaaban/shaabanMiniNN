import itertools
import numpy as np

class HyperparamterTuner:

    def __init__(self, model_fn, optimizer_fn, loss_fn, trainer_fn, param_grid, metric="loss"):
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.trainer_fn = trainer_fn
        self.param_grid = param_grid
        self.metric = metric

        self.result = []

    def _generate_configs(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            yield config
    

    def search(self, X_trina, y_train, X_val, y_val):
        best_score = np.inf
        best_config = None

        for config in self._generate_configs():

            model = self.model_fn(config)
            optimizer = self.optimizer_fn(config)
            trainer = self.trainer_fn(model, self.loss_fn, optimizer)

            trainer.fit(X_trina, y_train)

            val_loss = self._evaluate(model, X_val, y_val)

            self.result.append({
                "config": config,
                "val_loss": val_loss
            })

            if val_loss < best_score:
                best_score = val_loss
                best_config = config

        return best_config, best_score
    
    def _evaluate(self, model, X, y):
        model.eval()
        scores = model.forward(X)
        loss = self.loss_fn.forward(scores, y)
        return loss