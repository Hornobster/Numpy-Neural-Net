#!/usr/local/bin/python
import numpy as np
from layers import *
from nn import *

# constants
epsilon = 0.0001

'''
Tests the analytical gradient computed in the backward pass of the SoftmaxCrossEntropyLoss layer
against the numerical gradient
'''
def testSoftmaxCrossEntropyLoss():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.zeros((5, 5))
    l[:] = np.eye(5)

    # create softmax loss layer 
    softmax_loss = SoftmaxCrossEntropyLoss(X, L)
    softmax_loss.loss.grad = 1.0

    # get analytical gradient
    softmax_loss.forward()
    softmax_loss.backward()
    grad_x = softmax_loss.bottom.grad.copy()

    # compute numerical gradient
    perturb = np.zeros_like(x)
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(l.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            softmax_loss.forward()

            num_grad_x[i][j] = softmax_loss.loss.value

            x[i][j] = old_x - epsilon

            softmax_loss.forward()

            num_grad_x[i][j] -= softmax_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    #print('num grad', num_grad_x)
    #print('grad', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('SoftmaxCrossEntropyLoss: norm diff %f %s' % (norm_diff, ok))

testSoftmaxCrossEntropyLoss()

