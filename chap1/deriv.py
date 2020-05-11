from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[np.ndarray], np.ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]


def deriv(
    func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001
) -> np.ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def double(x: np.ndarray) -> np.ndarray:
    return 2 * x


def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.2 * x, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def chain_deriv2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    """
    assert len(chain) == 2, "This function requires 'Chain' objects of length 2"

    assert (
        input_range.ndim == 1
    ), "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    df1dx = deriv(f1, input_range)  # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))
    # Multiplying these quantities together at each point
    return df1dx * df2du


def chain_deriv3(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    """
    assert len(chain) == 3, "This function requires 'Chain' objects of length 2"

    assert (
        input_range.ndim == 1
    ), "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1_of_x = f1(input_range)
    f2_of_x = f2(f1_of_x)

    df3du = deriv(f3, f2_of_x)
    df2dx = deriv(f2, f1_of_x)
    df1dx = deriv(f1, input_range)
    # Multiplying these quantities together at each point
    return df1dx * df2dx * df3du


def multiple_inputs_add(
    x: np.ndarray, y: np.ndarray, sigma: Array_Function
) -> np.ndarray:
    assert x.shape == y.shape
    a = x + y
    return sigma(a)


def matmul_forward(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    N = np.dot(X, Y)
    return N


def matmul_backward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    dNdx = np.transpose(w, (1, 0))
    return dNdx


def matrix_forward_extra(
    X: np.ndarray, W: np.ndarray, sigma: Array_Function
) -> np.ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    return S


def matrix_function_backward_1(
    X: np.ndarray, W: np.ndarray, sigma: Array_Function
) -> np.ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    dSdN = deriv(sigma, N)
    dNdX = np.transpose(W, (1, 0))
    return np.dot(dSdN, dNdX)

def matrix_function_forward_sum(
    X: np.ndarray,
    W: np.ndarray,
    sigma: Array_Function
) -> float:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    return L

def matrix_function_backward_sum_1(
    X: np.ndarray,
    W: np.ndarray,
    sigma: Array_Function
) -> np.ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    dLdS = np.ones_like(S)
    dSdN = deriv(sigmoid, N)
    dLdN = dLdS * dSdN
    dNdX = np.transpose(W, (1, 0))
    dLdX = np.dot(dLdN, dNdX)
    return dLdX


if __name__ == "__main__":
    print(deriv(double, np.arange(10)))
    PLOT_RANGE = np.arange(-3, 3, 0.01)
    chain_1 = [np.square, sigmoid]
    chain_2 = [sigmoid, np.square]

    # plt.plot(PLOT_RANGE, sigmoid(np.square(PLOT_RANGE)))
    # plt.plot(PLOT_RANGE, chain_deriv2(chain_1, PLOT_RANGE))
        # plt.show()

    # plt.plot(PLOT_RANGE, np.square(sigmoid(PLOT_RANGE)))
    # plt.plot(PLOT_RANGE, chain_deriv2(chain_2, PLOT_RANGE))
    # plt.show()

    plt.plot(PLOT_RANGE, np.square(sigmoid(leaky_relu(PLOT_RANGE))))
    plt.plot(PLOT_RANGE, chain_deriv3([leaky_relu, sigmoid, np.square], PLOT_RANGE))
    plt.show()

    X = np.random.randn(2 ,3)
    W = np.random.randn(3, 2)
    print((matrix_forward_extra(X + np.array((0, 0, 0.01)), W, sigmoid) - matrix_forward_extra(X, W, sigmoid)) / 0.01)
    print(matrix_function_backward_1(X, W, sigmoid))
    print(matrix_function_backward_1(X + np.array((0, 0, 0.01)), W, sigmoid))

    L = matrix_function_forward_sum(X, W, sigmoid)
    dLdX = matrix_function_backward_sum_1(X, W, sigmoid)

    print()
    print()
    print()

    print("X", X)
    print("L", L)
    print("dLdX", dLdX)

    X1 = X.copy()
    X1[0, 0] += 0.001
    print(round(
    (matrix_function_forward_sum(X1, W, sigmoid) - \
    matrix_function_forward_sum(X, W, sigmoid)) / 0.001, 4))

    x11 = []
    y11 = []
    for i in np.arange(0, 2, 0.1):
        X1 = X.copy()
        X1[0, 0] += i
        x11.append(X1[0, 0])
        Y = matrix_function_forward_sum(X1, W, sigmoid)
        y11.append(Y)

    print()
    print(y11[-1] - y11[0])
    plt.plot(x11, y11)
    plt.show()