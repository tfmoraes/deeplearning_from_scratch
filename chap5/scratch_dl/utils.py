import numpy as np


def assert_same_shape(output: np.ndarray, output_grad: np.ndarray):
    assert (
        output.shape == output_grad.shape
    ), """
    Two ndarray should have the same shape; instead, first ndarray's shape is {0}
    and second ndarray's shape is {1}.
    """.format(
        tuple(output_grad.shape), tuple(output.shape)
    )
    return None


def assert_dim(t: np.ndarray, dim: int):
    assert (
        len(t.shape) == dim
    ), """
    Tensor expected to have dimension {0}, instead has dimension {1}
    """.format(
        dim, len(t.shape)
    )
    return None


def _pad_1d(inp: np.ndarray, num: int) -> np.ndarray:
    assert_dim(inp, 1)
    return np.pad(inp, num)


def _conv_1d(inp: np.ndarray, param: np.ndarray) -> np.ndarray:
    # assert correct dimensions
    assert_dim(inp, 1)
    assert_dim(param, 1)

    # pad the input
    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)

    out = np.zeros(inp.shape)

    # convolution
    for o in range(out.shape[0]):
        for p in range(param_len):
            out[o] += param[p] * input_pad[o + p]

    assert_same_shape(inp, out)
    return out


def _param_grad_1d(
    inp: np.ndarray, param: np.ndarray, output_grad: np.ndarray = None
) -> np.ndarray:

    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)

    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    # Zero padded 1 dimensional convolution
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            param_grad[p] += input_pad[o + p] * output_grad[o]

    assert_same_shape(param_grad, param)

    return param_grad


def _input_grad_1d(
    inp: np.ndarray, param: np.ndarray, output_grad: np.ndarray = None
) -> np.ndarray:

    param_len = param.shape[0]
    param_mid = param_len // 2
    inp_pad = _pad_1d(inp, param_mid)

    if output_grad is None:
        output_grad = np.ones_like(inp)
    else:
        assert_same_shape(inp, output_grad)

    output_pad = _pad_1d(output_grad, param_mid)

    # Zero padded 1 dimensional convolution
    param_grad = np.zeros_like(param)
    input_grad = np.zeros_like(inp)

    for o in range(inp.shape[0]):
        for f in range(param.shape[0]):
            input_grad[o] += output_pad[o + param_len - f - 1] * param[f]

    assert_same_shape(param_grad, param)

    return input_grad


def main():
    input_1d = np.array([1, 2, 3, 4, 5])
    param_1d = np.array([1, 1, 1])
    print(_conv_1d(input_1d, param_1d))
    print(_input_grad_1d(input_1d, param_1d))
    print(_param_grad_1d(input_1d, param_1d))


if __name__ == "__main__":
    main()
