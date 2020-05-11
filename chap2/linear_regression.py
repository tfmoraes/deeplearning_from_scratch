from typing import Callable, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[np.ndarray], np.ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]

def forward_linear_regression(
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    weights: Dict[str, np.ndarray]
) -> Tuple[float, Dict[str, np.ndarray]]:
    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights["W"].shape[0]
    assert weights["B"].shape[0] == weights["B"].shape[1] == 1
    N = np.dot(X_batch, weights["W"])
    P = N + weights["B"]
    loss = np.mean((y_batch - P)**2)

    # save the information computed on the forward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch
    return loss, forward_info


def loss_gradient(
    forward_info: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    batch_size = forward_info["X"].shape[0]
    dLdP = -2 * (forward_info["y"] - forward_info["P"])
    dPdN = np.ones_like(forward_info["N"])
    dPdB = np.ones_like(weights["B"])
    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info["X"], (1, 0))
    dLdW = np.dot(dNdW, dLdN)
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradient: Dict[str, np.ndarray] = {}
    loss_gradient["W"] = dLdW
    loss_gradient["B"] = dLdB
    return loss_gradient


def train(X, Y, learning_rate=0.001):
    w = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1, 1)
    weights = {}
    weights["W"] = w
    weights["B"] = b
    losses = []
    for i in range(1000):
        loss, forward_info = forward_linear_regression(X, Y, weights)
        losses.append(loss)
        if loss < 0.0001:
            break
        loss_grads = loss_gradient(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    print("It took {} steps and loss {}".format(i, loss))
    return weights, losses


def predict(
    X: np.ndarray,
    weights: Dict[str, np.ndarray]
) -> np.ndarray:
    N = np.dot(X, weights["W"])
    return N + weights["B"]


def mae(preds: np.array, actuals: np.array) -> float:
    "Compute mean absolute error"
    return np.mean(np.abs(preds - actuals))


def rmse(preds: np.array, actuals: np.array) -> float:
    "Compute root mean squared error"
    return np.sqrt(np.mean((preds - actuals)**2))


def main():
    # X = np.arange(-1, 1, 0.01).reshape(-1, 1)
    X = np.random.randn(100, 5)
    # Y = X**2 + 2*X + np.random.random()
    # Y = np.cos(X)
    Y = np.dot(X, np.random.random((5, 1))) + np.random.random((100, 1))
    weights, losses = train(X, Y)
    Yp = predict(X, weights)

    plt.scatter(range(Y.shape[0]), Y.flatten(), c='r', label="Y")
    plt.plot(range(Y.shape[0]), Yp.flatten(), c='g', label="linear_regression")
    plt.legend()
    plt.show()

    plt.plot(range(len(losses)), losses)
    plt.show()

    print("Mean absolute error:", mae(Yp, Y))
    print("Root mean squared error:", rmse(Yp, Y))
    print("Average distance:", rmse(Yp, Y) / Y.mean())

if __name__ == "__main__":
    main()