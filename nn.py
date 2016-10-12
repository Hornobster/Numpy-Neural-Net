class NeuralNet:
    def __init__(self):
        self.layers = {}
        self.fwd_layers = []
        self.params = []

    def init_params(self):
        for l in self.fwd_layers:
            l.init_params()

            l_params = l.get_params()
            if l_params is not None:
                self.params += l_params

    def get_params(self):
        return self.params

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

