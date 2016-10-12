import numpy as np

class Optimiser:
    def __init__(self, network):
        self.nn        = network
        self.step_sign = -1.0 # minimise by default

    def step(self):
        self.nn.forward()

        self.nn.reset_gradients()

        self.nn.backward()

        self.update_params()

    def update_params(self):
        pass

    def minimise(self):
        self.step_sign = -1.0
        return self

    def maximise(self):
        self.step_sign = 1.0
        return self

class GradientDescentOptimiser(Optimiser):
    def __init__(self, network, step_size):
        Optimiser.__init__(self, network)

        self.step_size = abs(step_size)

    def update_params(self):
        for param in self.nn.get_params():
            param.value += (self.step_sign * self.step_size) * param.grad


