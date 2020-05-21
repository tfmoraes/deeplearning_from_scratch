from typing import Callable, List, Tuple
import numpy as np

from .network import NeuralNetwork
from .optimizer import Optimizer

def calc_accuracy_model(y_pred, y_test):
    acc = np.equal(y_pred.argmax(1), y_test.argmax(1)).sum() * 100.0 / y_pred.shape[0]
    return acc

def permute_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

class Trainer:
    """
    Trains a neural network
    """

    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        setattr(self.optim, "net", net)

    def generate_batches(
        self, X: np.ndarray, y: np.ndarray, size: int = 32
    ) -> Tuple[np.ndarray]:
        """
        Generates batches for training 
        """
        assert (
            X.shape[0] == y.shape[0]
        ), """
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        """.format(
            X.shape[0], y.shape[0]
        )

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]
            yield X_batch, y_batch

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        restart: bool = True,
    ) -> None:
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

        self.optim.max_epochs = epochs
        self.optim._setup_decay()
        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            test_preds = self.net.predict(X_test)
            loss = self.net.loss.forward(test_preds, y_test)
            acc = calc_accuracy_model(test_preds, y_test)

            if self.optim.final_lr:
                self.optim._decay_lr()

            print(
                f"Epoch: {e} - loss: {loss:.3f}, acc: {acc:.3f}, lr: {self.optim.lr:.3f}"
            )
