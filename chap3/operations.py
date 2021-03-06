from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


def permute_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


class Operation:
    """
    Base class for an operation in neural network
    """

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """
        Store input_ and apply it in self._output
        """
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        calls self._input_grad() function. Checks that the apropriate shapes matches
        """
        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad)
        assert self.input_.shape == self.input_grad.shape
        return self.input_grad

    def _output(self) -> np.ndarray:
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

    def _output(self) -> np.ndarray:
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

    def _output(self) -> np.ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Linear(Operation):
    """
    "Identity" activation function
    """

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self) -> np.ndarray:
        """Pass through"""
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Pass through"""
        return output_grad


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

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """
        Pass input throuhg a serie of operations
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
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

    def __init__(self, neurons: int, activation: Operation = Sigmoid()) -> None:
        super().__init__(neurons)
        self.activation = activation
        self.seed = None

    def _setup_layer(self, input_: np.ndarray) -> None:
        """
        Defines the operations of a fully connected layer
        """
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # Weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # Bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [
            WeighMutiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]


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

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Pass data forward through a serie of layer
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Pass data backward through a serie of layer
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Passes data forward through the layers. Compute the loss. Passes data backward through the layer
        """
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        """
        Get the params from network
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads


class Optimizer:
    """
    Base class for optimizer
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        pass


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


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

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                print(f"Validation loss after {e+1} epochs is {loss:.3f}")


def main():
    #X = np.random.randn(100, 5)
    # X = np.random.randn(1000).reshape(-1, 1)
    #Y = np.dot(X, np.random.random((5, 1))) + np.random.random((100, 1))
    # Y = X**2 + 2*X + np.random.random()
    X, Y = load_boston(return_X_y=True)
    Y.shape = (-1, 1)
    s = StandardScaler()
    X = s.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    optimizer = SGD(lr=0.01)
    neural_network = NeuralNetwork(
        layers=[
            Dense(neurons=13, activation=Sigmoid()),
            Dense(neurons=13, activation=Sigmoid()),
            Dense(neurons=1, activation=Linear()),
        ],
        loss=MeanSquaredError(),
    )
    trainer = Trainer(neural_network, optimizer)
    trainer.fit(X_train, y_train, X_test, y_test, epochs=200, eval_every=10, seed=42)

    X_proof = X #np.arange(-20, 20, 0.01).reshape(-1, 1)
    y_proof = neural_network.forward(X_proof)

    plt.scatter(X[:, 0], Y.flatten(), c='r', label="Y")
    plt.scatter(X_proof[:, 0], y_proof.flatten(), c='g', label="linear_regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
