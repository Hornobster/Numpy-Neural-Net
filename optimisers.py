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

class GradientDescentMomentumOptimiser(Optimiser):
    def __init__(self, network, step_size, momentum = 0.9):
        Optimiser.__init__(self, network)

        self.step_size = abs(step_size)
        self.momentum  = momentum

        # initialise variables for momentum
        self.last_param_updates = []
        for param in self.nn.get_params():
            self.last_param_updates.append(np.zeros_like(param.value))

    def update_params(self):
        for param, last_update in zip(self.nn.get_params(), self.last_param_updates):
            update          = self.momentum * last_update + self.step_size * param.grad
            param.value    += self.step_sign * update
            last_update[:]  = update

