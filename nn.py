class NeuralNet:
    def __init__(self):
        self.layers = {}
        self.fwd_layers = []

    def init_params(self):
        for l in self.fwd_layers:
            l.init_params()

    def update_params(self, step_size):
        for l in self.fwd_layers:
            l.update_params(step_size)

    def reset_gradients(self):
        for l in self.fwd_layers:
            l.reset_gradient()

    def forward(self):
        for l in self.fwd_layers:
            l.forward()

    def backward(self):
        for l in reversed(self.fwd_layers):
            l.backward()

    def add(self, name, l):
        self.layers[name] = l

        self.fwd_layers.append(l)

        return l

