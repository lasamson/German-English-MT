""" A wrapper class for the optimizer for the transformer """
import numpy as np


class ScheduledOptimizer():
    """ 
    A simple wrapper class for learning rate scheduling.
    Vary the learning rate over the course of traning.
    Increase the learning rate linearly for `n_warmup_steps`
    training steps and decreasing it proportionally to the inverse
    square root of the step number
    """

    def __init__(self, optimizer, d_model, factor=2, n_warmup_steps=4000):
        self._optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.factor = factor
        self.n_current_steps = 0

    def step_and_update_lr(self):
        """ Step with the inner optimizer """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """ Zero out the gradient by the inner optimizer """
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        """ Get the learning rate scale """
        return self.factor * (self.d_model ** (-.5) * np.min([np.power(self.n_current_steps, -0.5),
                                                              np.power(self.n_warmup_steps, -1.5) * self.n_current_steps]))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_current_steps += 1

        # get the new learning rate at the current step
        lr = self._get_lr_scale()

        # update the learning rate every step
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
