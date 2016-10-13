import numpy as np

class Regulariser:
    def __init__(self, factor = 1.0):
        self.factor = factor

    def get_gradient(self, weights):
        pass

class L1Regulariser(Regulariser):
    def get_gradient(self, weights):
        return np.sign(weights) * self.factor

class L2Regulariser(Regulariser):
    def get_gradient(self, weights):
        return self.factor * weights
