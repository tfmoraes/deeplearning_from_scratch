import pickle
from typing import Callable, List, Tuple

import numpy as np

from .layers import Layer
from .loss import Loss


class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1):
        """
        Need layers and loss
        """
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: np.ndarray, inference: bool = False) -> np.ndarray:
        """
        Pass data forward through a serie of layer
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out, inference)
        return x_out

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Pass data backward through a serie of layer
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, inference: bool = False) -> float:
        """
        Passes data forward through the layers. Compute the loss. Passes data backward through the layer
        """
        predictions = self.forward(x_batch, inference)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def predict(self, X):
        return self.forward(X, True)

    def params(self):
        """
        Get the params from network
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> None:
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            return tmp_dict
