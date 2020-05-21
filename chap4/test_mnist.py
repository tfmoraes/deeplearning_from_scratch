import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scratch_dl.layers import Dense
from scratch_dl.loss import SoftmaxCrossEntropyLoss
from scratch_dl.network import NeuralNetwork
from scratch_dl.operations import Linear, Sigmoid, Tanh
from scratch_dl.optimizer import SGDMomentun
from scratch_dl.train import Trainer


def load_mnist():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    return X, y


def main():
    X, Y = load_mnist()
    X = X / 255.0
    Y = Y.reshape(-1, 1)

    oe = OneHotEncoder()
    oe.fit(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
    X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

    optimizer = SGDMomentun(lr=0.15, momentun=0.9, final_lr=0.05, decay_type="linear")
    neural_network = NeuralNetwork(
        layers=[
            Dense(neurons=89, activation=Tanh(), dropout=0.8, weight_init="glorot"),
            Dense(neurons=10, activation=Linear(), weight_init="glorot"),
        ],
        loss=SoftmaxCrossEntropyLoss(),
    )
    trainer = Trainer(neural_network, optimizer)
    trainer.fit(
        X_train,
        oe.transform(y_train).toarray(),
        X_test,
        oe.transform(y_test).toarray(),
        epochs=50,
        eval_every=10,
        seed=42,
    )


if __name__ == "__main__":
    main()
