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
    print('%s SoftmaxCrossEntropyLoss: norm diff %f' % (ok, norm_diff))

'''
Tests the analytical gradient computed in the backward pass of the MSELoss layer
against the numerical gradient
'''
def testMSELoss():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.zeros((5, 5))
    l[:] = np.eye(5)

    # create mse loss layer
    mse_loss = MSELoss(X, L)
    mse_loss.loss.grad = 1.0

    # get analytical gradient
    mse_loss.forward()
    mse_loss.backward()
    grad_x = mse_loss.bottom.grad.copy()

    # compute numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(l.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            mse_loss.forward()

            num_grad_x[i][j] = mse_loss.loss.value

            x[i][j] = old_x - epsilon

            mse_loss.forward()

            num_grad_x[i][j] -= mse_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    #print('num grad', num_grad_x)
    #print('grad', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s MSELoss: norm diff %f' % (ok, norm_diff))

'''
Tests the analytical gradient computed in the backward pass of the CrossEntropyLoss layer
against the numerical gradient
'''
def testCrossEntropyLoss():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.ones((5, 5))
    l[:] = np.eye(5)

    # create cross entropy loss layer
    crossentropy_loss = CrossEntropyLoss(X, L)
    crossentropy_loss.loss.grad = 1.0

    # get analytical gradient
    crossentropy_loss.forward()
    crossentropy_loss.backward()
    grad_x = crossentropy_loss.bottom.grad.copy()

    # compute numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(l.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            crossentropy_loss.forward()

            num_grad_x[i][j] = crossentropy_loss.loss.value

            x[i][j] = old_x - epsilon

            crossentropy_loss.forward()

            num_grad_x[i][j] -= crossentropy_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    # print('num grad', num_grad_x)
    # print('grad', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s CrossEntropyLoss: norm diff %f' % (ok, norm_diff))

'''
Tests the analytical gradient computed in the backward pass of the ReLu layer
against the numerical gradient
'''
def testReLu():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.random.rand(5, 5) - 0.5 # random values between -0.5 and 0.5
    l[:] = np.eye(5)

    # create relu and mse loss layers
    relu = ReLu(X)
    mse_loss = MSELoss(relu, L)
    mse_loss.loss.grad = 1.0

    # get analytical gradient
    relu.forward()
    mse_loss.forward()
    mse_loss.backward()
    relu.backward()
    grad_x = relu.bottom.grad.copy()

    # compute numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(l.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            relu.forward()
            mse_loss.forward()

            num_grad_x[i][j] = mse_loss.loss.value

            x[i][j] = old_x - epsilon

            relu.forward()
            mse_loss.forward()

            num_grad_x[i][j] -= mse_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    # print('num grad', num_grad_x)
    # print('grad', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s ReLu: norm diff %f' % (ok, norm_diff))

testSoftmaxCrossEntropyLoss()
testMSELoss()
testCrossEntropyLoss()
testReLu()

