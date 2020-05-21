from typing import Callable, List, Tuple
import numpy as np

from .operations import WeighMutiply, BiasAdd, Dropout, Sigmoid, Linear, Tanh, Operation, ParamOperation

class Layer:
    def __init__(self, neurons: int):
        """
        The number of neurons roughly corresponds to the "breadth" of the layer
        """
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation]

    def _setup_layer(self, num_in: int) -> None:
        raise NotImplementedError

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        """
        Pass input throuhg a serie of operations
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_, inference)
        self.output = input_
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Passes output_grad through a serie of operations
        """
        assert self.output.shape == output_grad.shape
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grads = output_grad
        self._param_grads()
        return input_grads

    def _param_grads(self) -> None:
        """
        Extract param_grads from layer's operations
        """
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:
        """
        Extract params from layer's operations
        """
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """
    A fully connected layer
    """

    def __init__(
        self,
        neurons: int,
        activation: Operation = Sigmoid(),
        dropout: float = 1.0,
        weight_init: str = "standard",
    ) -> None:
        super().__init__(neurons)
        self.activation = activation
        self.seed = None
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: np.ndarray) -> None:
        """
        Defines the operations of a fully connected layer
        """
        if self.seed:
            np.random.seed(self.seed)

        num_in = input_.shape[1]
        if self.weight_init == "glorot":
            scale = 2.0 / (num_in + self.neurons)
        else:
            scale = 1.0

        self.params = []

        # Weights
        self.params.append(
            np.random.normal(loc=0, scale=scale, size=(num_in, self.neurons))
        )

        # Bias
        self.params.append(np.random.normal(loc=0, scale=scale, size=(1, self.neurons)))

        self.operations = [
            WeighMutiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))
