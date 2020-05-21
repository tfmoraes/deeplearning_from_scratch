from typing import Callable, List, Tuple
import numpy as np
from scipy.special import logsumexp

def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

class Loss:
    """
    The loss of a neural network
    """

    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert prediction.shape == target.shape

        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def backward(self) -> np.ndarray:
        self.input_grad = self._input_grad()
        assert self.input_grad.shape == self.prediction.shape
        return self.input_grad

    def _output(self) -> float:
        """
        Every subclass of Loss must implement it
        """
        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        """
        Every subclass of Loss must implement it
        """
        raise NotImplementedError()


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def _output(self) -> float:
        loss = np.sum((self.prediction - self.target) ** 2) / self.prediction.shape[0]
        return loss

    def _input_grad(self) -> np.ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.single_output = False

    def _output(self) -> float:
        softmax_preds = softmax(self.prediction, axis=1)
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)
        softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (
            1.0 - self.target
        ) * np.log(1.0 - self.softmax_preds)
        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:
        return (self.softmax_preds - self.target) / self.prediction.shape[0]
