from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[np.ndarray], np.ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def forward_loss(
    X: np.ndarray, y: np.ndarray, weights: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], float]:
    M1 = np.dot(X, weights["W1"])
    N1 = M1 + weights["B1"]
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights["W2"])
    P = M2 + weights["B2"]
    loss = np.mean((y - P) ** 2)

    # save the information computed on the forward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info["X"] = X
    forward_info["M1"] = M1
    forward_info["N1"] = N1
    forward_info["O1"] = O1
    forward_info["M2"] = M2
    forward_info["P"] = P
    forward_info["y"] = y
    return forward_info, loss


def loss_gradients(
    forward_info: Dict[str, np.ndarray], weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Compute the partial derivatives of the loss with respect to each of the parameters in the neural network.
    """
    dLdP = -(forward_info["y"] - forward_info["P"])
    dPdM2 = np.ones_like(forward_info["M2"])
    dLdM2 = dLdP * dPdM2
    dPdB2 = np.ones_like(weights["B2"])
    dLdB2 = (dLdP * dPdB2).sum(axis=0)
    dM2dW2 = np.transpose(forward_info["O1"], (1, 0))
    dLdW2 = np.dot(dM2dW2, dLdP)
    dM2dO1 = np.transpose(weights["W2"], (1, 0))
    dLdO1 = np.dot(dLdM2, dM2dO1)
    dO1dN1 = sigmoid(forward_info["N1"]) * (1 - sigmoid(forward_info["N1"]))
    dLdN1 = dLdO1 * dO1dN1
    dN1dB1 = np.ones_like(weights["B1"])
    dN1dM1 = np.ones_like(forward_info["M1"])
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
    dLdM1 = dLdN1 * dN1dM1
    dM1dW1 = np.transpose(forward_info["X"], (1, 0))
    dLdW1 = np.dot(dM1dW1, dLdM1)

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients["W2"] = dLdW2
    loss_gradients["B2"] = dLdB2.sum(axis=0)
    loss_gradients["W1"] = dLdW1
    loss_gradients["B1"] = dLdB1.sum(axis=0)

    return loss_gradients


def init_weights(input_size: int, hidden_size: int) -> Dict[str, np.ndarray]:
    weights: Dict[str, ndarray] = {}
    weights['W1'] = np.random.randn(input_size, hidden_size)
    weights['B1'] = np.random.randn(1, hidden_size)
    weights['W2'] = np.random.randn(hidden_size, 1)
    weights['B2'] = np.random.randn(1, 1)
    return weights


def train(X, Y, learning_rate=0.001, hidden_size=13):
    weights = init_weights(X.shape[1], hidden_size)
    losses = []
    for i in range(1000):
        forward_info, loss = forward_loss(X, Y, weights)
        losses.append(loss)
        if loss < 0.0001:
            break
        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    print("It took {} steps and loss {}".format(i, loss))
    return weights, losses


def predict(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    M1 = np.dot(X, weights["W1"])
    N1 = M1 + weights["B1"]
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights["W2"])
    P = M2 + weights["B2"]
    return P


def mae(preds: np.array, actuals: np.array) -> float:
    "Compute mean absolute error"
    return np.mean(np.abs(preds - actuals))


def rmse(preds: np.array, actuals: np.array) -> float:
    "Compute root mean squared error"
    return np.sqrt(np.mean((preds - actuals) ** 2))


def main():
    s = StandardScaler()
    #boston = load_boston()
    #X = boston.data
    #X = s.fit_transform(X)
    #Y = boston.target.reshape(-1, 1)
    X = np.arange(-3, 3, 0.01).reshape(-1, 1)
    #X = np.random.randn(100, 5)
    #Y = X**2 + 2*X + np.random.random()
    Y = np.cos(2*X)
    #Y = np.dot(X, np.random.random((5, 1))) + np.random.random((100, 1))
    weights, losses = train(X, Y, hidden_size=13)
    Yp = predict(X, weights)

    plt.scatter(range(Y.shape[0]), Y.flatten(), c="r", label="Y")
    plt.plot(range(Y.shape[0]), Yp.flatten(), c="g", label="linear_regression")
    plt.legend()
    plt.show()

    plt.plot(range(len(losses)), losses)
    plt.show()

    print("Mean absolute error:", mae(Yp, Y))
    print("Root mean squared error:", rmse(Yp, Y))
    print("Average distance:", rmse(Yp, Y) / Y.mean())


if __name__ == "__main__":
    main()
