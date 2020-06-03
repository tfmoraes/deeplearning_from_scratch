from typing import Callable, List, Tuple
import numpy as np

class Optimizer:
    """
    Base class for optimizer
    """

    def __init__(
        self, lr: float = 0.01, final_lr: float = 0.0, decay_type="exponential"
    ):
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True
        self.max_epochs = 100

    def _setup_decay(self) -> None:
        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(
                self.final_lr / self.lr, 1.0 / (self.max_epochs - 1)
            )
        elif self.decay_type == "linear":
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

        print("Decay per epoch", self.decay_per_epoch)

    def _decay_lr(self) -> None:
        if not self.decay_per_epoch:
            return
        elif self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch
        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

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


class SGDMomentun(Optimizer):
    """
    SGD with momentun.
    """

    def __init__(self, lr: float = 0.01, momentun: float = 0.9, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)
        self.momentun = momentun

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(
            self.net.params(), self.net.param_grads(), self.velocities
        ):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:
        # Update velocity
        kwargs["velocity"] *= self.momentun
        kwargs["velocity"] += self.lr * kwargs["grad"]
        kwargs["param"] -= kwargs["velocity"]
