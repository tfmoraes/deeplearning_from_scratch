from typing import Callable, List, Tuple

import numpy as np


class Operation:
    """
    Base class for an operation in neural network
    """

    def __init__(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle input_, output and input_grad
        try:
            del state["input_"]
        except KeyError:
            pass
        try:
            del state["output"]
        except KeyError:
            pass
        try:
            del state["input_grad"]
        except KeyError:
            pass
        return state

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        """
        Store input_ and apply it in self._output
        """
        self.input_ = input_
        self.output = self._output(inference)
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        calls self._input_grad() function. Checks that the apropriate shapes matches
        """
        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad)
        assert self.input_.shape == self.input_grad.shape
        return self.input_grad

    def _output(self, inference: bool) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    Operation with parameters
    """

    def __init__(self, param: np.ndarray):
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert self.input_.shape == self.input_grad.shape
        assert self.param.shape == self.param_grad.shape
        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class WeighMutiply(ParamOperation):
    """
    Weight multiplication operation for neural network
    """

    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def _output(self, inference: bool) -> np.ndarray:
        """
        Compute output
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Compute input gradient
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Compute parameter gradient
        """
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    Compute Bia addition
    """

    def __init__(self, B: np.ndarray):
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self, inference: bool) -> np.ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Tanh(Operation):
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        tanh_backward = 1.0 - self.output * self.output
        input_grad = tanh_backward * output_grad
        return input_grad


class Linear(Operation):
    """
    "Identity" activation function
    """

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        """Pass through"""
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Pass through"""
        return output_grad


class Dropout(Operation):
    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool) -> np.ndarray:
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask
