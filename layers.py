import numpy as np

class Blob:
    def __init__(self, data):
        self.value = data
        self.grad  = np.zeros_like(self.value)
        self.shape = data.shape

    def reset_gradient(self):
        self.grad[:] = 0.0

class Variable:
    def __init__(self, data):
        self.value = data
        self.grad  = np.zeros_like(data)

class Layer:
    def init_params(self):
        pass

    def get_params(self):
        return None

    def reset_gradient(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class Input(Layer):
    def __init__(self, batch_size, input_size):
        self.top = Blob(np.zeros((batch_size, input_size)))

class InnerProduct(Layer):
    def __init__(self, prev_layer, num_outputs):
        self.bottom      = prev_layer.top
        # num rows in bottom is the batch size
        # num columns in bottom is each input size
        self.batch_size  = self.bottom.value.shape[0]
        self.num_inputs  = self.bottom.value.shape[1]
        self.num_outputs = num_outputs
        self.top         = Blob(np.zeros((self.batch_size, self.num_outputs)))

    def init_params(self):
        self.weights = Variable(np.random.rand(self.num_inputs, self.num_outputs) - 0.5)
        self.bias    = Variable(np.full(self.num_outputs, 0.1))

    def get_params(self):
        return [self.weights, self.bias]

    def reset_gradient(self):
        self.top.reset_gradient()
        self.weights.grad[:] = 0.0
        self.bias.grad[:]    = 0.0
    
    def forward(self):
        self.top.value = np.dot(self.bottom.value, self.weights.value) + self.bias.value

    def backward(self):
        self.bottom.grad += np.dot(self.top.grad, self.weights.value.transpose())

        for i in range(self.batch_size):
            self.weights.grad += np.dot(self.bottom.value[i, np.newaxis].transpose(), self.top.grad[i, np.newaxis])

        self.bias.grad += np.sum(self.top.grad, axis=0)

class LinearInterpolation(Layer):
    def __init__(self, prev_layer1, prev_layer2):
        self.bottom1 = prev_layer1.top
        self.bottom2 = prev_layer2.top
        # check if input layers have same shape
        assert(self.bottom1.value.shape == self.bottom2.value.shape)

        self.batch_size = self.bottom1.shape[0]
        self.num_inputs = self.bottom1.shape[1]
        self.top        = Blob(np.zeros_like(self.bottom1.value))

    def init_params(self):
        self.a      = Variable(0.5)

    def get_params(self):
        return [self.a]

    def reset_gradient(self):
        self.top.reset_gradient()
        self.a.grad = 0.0

    def forward(self):
        self.top.value = (1.0 - self.a.value) * self.bottom1.value + self.a.value * self.bottom2.value

    def backward(self):
        self.bottom1.grad += (1.0 - self.a.value) * self.top.grad
        self.bottom2.grad += self.a.value * self.top.grad
        self.a.grad        = np.sum(np.multiply((self.bottom2.value - self.bottom1.value), self.top.grad))

class Sin(Layer):
    def __init__(self, prev_layer):
        self.bottom = prev_layer.top
        self.top    = Blob(np.zeros_like(self.bottom.value))

    def reset_gradient(self):
        self.top.reset_gradient()

    def forward(self):
        self.top.value = np.sin(self.bottom.value)

    def backward(self):
        self.bottom.grad += np.multiply(np.cos(self.bottom.value), self.top.grad)

class ReLu(Layer):
    def __init__(self, prev_layer):
        self.bottom = prev_layer.top
        self.top    = Blob(np.zeros_like(self.bottom.value))

    def reset_gradient(self):
        self.top.reset_gradient()

    def forward(self):
        self.top.value = np.maximum(0, self.bottom.value)

    def backward(self):
        self.bottom.grad += np.multiply((self.bottom.value > 0), self.top.grad)

class Sigmoid(Layer):
    def __init__(self, prev_layer):
        self.bottom = prev_layer.top
        self.top    = Blob(np.zeros_like(self.bottom.value))

    def reset_gradient(self):
        self.top.reset_gradient()

    def forward(self):
        self.top.value = 1.0 / (np.exp(-self.bottom.value) + 1)

    def backward(self):
        self.bottom.grad += np.multiply(np.multiply(1 - self.top.value, self.top.value), self.top.grad)

class Softmax(Layer):
    def __init__(self, prev_layer):
        self.bottom      = prev_layer.top
        self.batch_size  = self.bottom.shape[0]
        self.num_inputs  = self.bottom.shape[1]
        self.top         = Blob(np.zeros_like(self.bottom.value))

    def reset_gradient(self):
        self.top.reset_gradient()

    def forward(self):
        exp_bottom     = np.exp(self.bottom.value)
        self.top.value = exp_bottom / np.sum(exp_bottom, axis = 1)[:, np.newaxis]

    def backward(self):
        # there's probably a vectorized way of doing this...
        # J_i,j = top_i * (kronecker_i,j - top_j)
        jacobian = np.zeros((self.num_inputs, self.num_inputs))
        for x in range(self.batch_size):
            for i in range(self.num_inputs):
                for j in range(self.num_inputs):
                    if i == j:
                        jacobian[i][j] = self.top.value[x][i] * (1 - self.top.value[x][i])
                    else:
                        jacobian[i][j] = - self.top.value[x][i] * self.top.value[x][j]

            self.bottom.grad[x] += np.dot(self.top.grad[x], jacobian)

class Tanh(Layer):
    def __init__(self, prev_layer):
        self.bottom = prev_layer.top
        self.top    = Blob(np.zeros_like(self.bottom.value))

    def reset_gradient(self):
        self.top.reset_gradient()

    def forward(self):
        exp2 = np.exp(2 * self.bottom.value)
        self.top.value = (exp2 - 1) / (exp2 + 1)

    def backward(self):
        self.bottom.grad += np.multiply(1 - np.square(self.top.value), self.top.grad)

class SoftmaxCrossEntropyLoss(Layer):
    def __init__(self, prev_layer, labels_layer):
        self.bottom     = prev_layer.top
        self.labels     = labels_layer.top
        self.batch_size = self.bottom.value.shape[0]
        self.num_inputs = self.bottom.value.shape[1]
        self.top        = Blob(np.zeros_like(self.bottom.value))
        self.loss       = Blob(np.zeros(1))

    def reset_gradient(self):
        self.top.reset_gradient()

    def cross_entropy(self, i, l):
        return -np.sum(np.multiply(np.log(i), l), axis = 1)

    def forward(self):
        exp_bottom      = np.exp(self.bottom.value)
        self.top.value  = exp_bottom / np.sum(exp_bottom, axis = 1)[:, np.newaxis]
        self.loss.value = np.mean(self.cross_entropy(self.top.value, self.labels.value))

    def backward(self):
        labels_sum = np.sum(self.labels.value, axis=1, keepdims=True)
        self.bottom.grad += (np.multiply(self.top.value, labels_sum) - self.labels.value) / self.batch_size

class MSELoss(Layer):
    def __init__(self, prev_layer, labels_layer):
        self.bottom     = prev_layer.top
        self.labels     = labels_layer.top
        self.batch_size = self.bottom.value.shape[0]
        self.num_inputs = self.bottom.value.shape[1]
        self.loss       = Blob(np.zeros(1))

    def forward(self):
        self.loss.value = np.mean(np.square(self.bottom.value - self.labels.value))

    def backward(self):
        self.bottom.grad += (2.0 / (self.num_inputs * self.batch_size)) * (self.bottom.value - self.labels.value)

class CrossEntropyLoss(Layer):
    def __init__(self, prev_layer, labels_layer):
        self.bottom     = prev_layer.top
        self.labels     = labels_layer.top
        self.batch_size = self.bottom.value.shape[0]
        self.num_inputs = self.bottom.value.shape[1]
        self.loss       = Blob(np.zeros(1))

    def cross_entropy(self, i, l):
        return -np.sum(np.multiply(np.log(i), l), axis = 1)

    def forward(self):
        self.loss.value = np.mean(self.cross_entropy(self.bottom.value, self.labels.value))

    def backward(self):
        self.bottom.grad += - (self.labels.value / self.bottom.value) / self.batch_size

