import numpy as np

class Optimiser:
    def __init__(self, network):
        self.nn        = network
        self.step_sign = -1.0 # minimise by default

    def before_forward(self):
        pass

    def before_backward(self):
        pass

    def step(self):
        self.before_forward()

        self.nn.forward()

        self.nn.reset_gradients()

        self.before_backward()

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
        self.last_params_updates = []
        for param in self.nn.get_params():
            self.last_params_updates.append(np.zeros_like(param.value))

    def update_params(self):
        for param, last_update in zip(self.nn.get_params(), self.last_params_updates):
            update          = self.momentum * last_update + self.step_size * param.grad
            param.value    += self.step_sign * update
            last_update[:]  = update

class NesterovAcceleratedGradientOptimiser(Optimiser):
    def __init__(self, network, step_size, momentum = 0.9):
        Optimiser.__init__(self, network)

        self.step_size = abs(step_size)
        self.momentum = momentum

        # initialise variables for momentum
        self.last_params_updates = []
        for param in self.nn.get_params():
            self.last_params_updates.append(np.zeros_like(param.value))

    def before_forward(self):
        for param, last_update in zip(self.nn.get_params(), self.last_params_updates):
            param.value += (self.step_sign * self.momentum) * last_update

    def update_params(self):
        for param, last_update in zip(self.nn.get_params(), self.last_params_updates):
            update          = self.momentum * last_update + self.step_size * param.grad
            # add only the second update factor, since we've already added the first factor in before_forward()
            param.value    += (self.step_sign * self.step_size) * param.grad
            last_update[:]  = update

class AdagradOptimiser(Optimiser):
    def __init__(self, network, step_size = 0.01, epsilon = 1e-8):
        Optimiser.__init__(self, network)

        self.step_size = abs(step_size)
        self.epsilon   = epsilon

        # initialise accumulated square of gradients
        self.gradients_acc_square = []
        for param in self.nn.get_params():
            self.gradients_acc_square.append(np.zeros_like(param.grad))

    def update_params(self):
        for param, acc_square in zip(self.nn.get_params(), self.gradients_acc_square):
            acc_square  += np.square(param.grad)
            param.value += self.step_sign * np.multiply(param.grad, self.step_size / np.sqrt(acc_square + self.epsilon))

class AdaDeltaOptimiser(Optimiser):
    def __init__(self, network, decay_rate = 0.9, epsilon = 1e-8):
        Optimiser.__init__(self, network)

        self.decay_rate = decay_rate
        self.epsilon = epsilon

        # initialise accumulated square of gradients and updates
        self.gradients_acc_square = []
        self.updates_acc_square = []
        for param in self.nn.get_params():
            self.gradients_acc_square.append(np.zeros_like(param.grad))
            self.updates_acc_square.append(np.zeros_like(param.value))

    def update_params(self):
        for param, gradient_acc_square, update_acc_square in zip(self.nn.get_params(), self.gradients_acc_square, self.updates_acc_square):
            # accumulate gradient
            gradient_acc_square *= self.decay_rate
            gradient_acc_square += (1.0 - self.decay_rate) * np.square(param.grad)

            # compute update
            update = np.multiply(np.sqrt(update_acc_square + self.epsilon) / np.sqrt(gradient_acc_square + self.epsilon), param.grad)

            # accumulate update
            update_acc_square *= self.decay_rate
            update_acc_square += (1.0 - self.decay_rate) * np.square(update)

            # apply update
            param.value += self.step_sign * update

