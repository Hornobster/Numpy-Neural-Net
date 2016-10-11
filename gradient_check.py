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
        for j in range(x.shape[1]):
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
        for j in range(x.shape[1]):
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
        for j in range(x.shape[1]):
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
        for j in range(x.shape[1]):
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

'''
Tests the analytical gradient computed in the backward pass of the Sin layer
against the numerical gradient
'''
def testSin():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.random.rand(5, 5) - 0.5 # random values between -0.5 and 0.5
    l[:] = np.eye(5)

    # create sin and mse loss layers
    sin = Sin(X)
    mse_loss = MSELoss(sin, L)
    mse_loss.loss.grad = 1.0

    # get analytical gradient
    sin.forward()
    mse_loss.forward()
    mse_loss.backward()
    sin.backward()
    grad_x = sin.bottom.grad.copy()

    # compute numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            sin.forward()
            mse_loss.forward()

            num_grad_x[i][j] = mse_loss.loss.value

            x[i][j] = old_x - epsilon

            sin.forward()
            mse_loss.forward()

            num_grad_x[i][j] -= mse_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    # print('num grad', num_grad_x)
    # print('grad', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s Sin: norm diff %f' % (ok, norm_diff))

'''
Tests the analytical gradient computed in the backward pass of the Softmax layer
against the numerical gradient
'''
def testSoftmax():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.random.rand(5, 5) - 0.5 # random values between -0.5 and 0.5
    l[:] = np.eye(5)

    # create softmax and mse loss layers
    softmax = Softmax(X)
    mse_loss = MSELoss(softmax, L)
    mse_loss.loss.grad = 1.0

    # get analytical gradient
    softmax.forward()
    mse_loss.forward()
    mse_loss.backward()
    softmax.backward()
    grad_x = softmax.bottom.grad.copy()

    # compute numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            softmax.forward()
            mse_loss.forward()

            num_grad_x[i][j] = mse_loss.loss.value

            x[i][j] = old_x - epsilon

            softmax.forward()
            mse_loss.forward()

            num_grad_x[i][j] -= mse_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    # print('num grad', num_grad_x)
    # print('grad', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s Softmax: norm diff %f' % (ok, norm_diff))

'''
Tests the analytical gradient computed in the backward pass of the InnerProduct layer
against the numerical gradient
'''
def testInnerProduct():
    # create input layers
    X = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    l = L.top.value
    x[:] = np.random.rand(5, 5) - 0.5 # random values between -0.5 and 0.5
    l[:] = np.eye(5)

    # create inner product and mse loss layers
    ip = InnerProduct(X, 5)
    mse_loss = MSELoss(ip, L)
    mse_loss.loss.grad = 1.0

    ip.init_params()

    # get analytical gradient
    ip.forward()
    mse_loss.forward()
    mse_loss.backward()
    ip.backward()
    grad_x = ip.bottom.grad.copy()
    grad_w = ip.w_grad.copy()
    grad_b = ip.b_grad.copy()

    # compute inputs X numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            ip.forward()
            mse_loss.forward()

            num_grad_x[i][j] = mse_loss.loss.value

            x[i][j] = old_x - epsilon

            ip.forward()
            mse_loss.forward()

            num_grad_x[i][j] -= mse_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    # compute weights W numerical gradient
    w = ip.weights
    num_grad_w = np.zeros_like(w)

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            old_w = w[i][j]

            w[i][j] = old_w + epsilon

            ip.forward()
            mse_loss.forward()

            num_grad_w[i][j] = mse_loss.loss.value

            w[i][j] = old_w - epsilon

            ip.forward()
            mse_loss.forward()

            num_grad_w[i][j] -= mse_loss.loss.value
            num_grad_w[i][j] /= 2.0 * epsilon

            w[i][j] = old_w

    # compute bias B numerical gradient
    b = ip.bias
    num_grad_b = np.zeros_like(b)

    for i in range(b.shape[0]):
        old_b = b[i]

        b[i] = old_b + epsilon

        ip.forward()
        mse_loss.forward()

        num_grad_b[i] = mse_loss.loss.value

        b[i] = old_b - epsilon

        ip.forward()
        mse_loss.forward()

        num_grad_b[i] -= mse_loss.loss.value
        num_grad_b[i] /= 2.0 * epsilon

        b[i] = old_b

    # inputs X
    # print('num grad x', num_grad_x)
    # print('grad x', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s InnerProduct X: norm diff %f' % (ok, norm_diff))

    # weights W
    # print('num grad w', num_grad_w)
    # print('grad w', grad_w)
    norm_diff = np.linalg.norm(grad_w - num_grad_w) / np.linalg.norm(grad_w + num_grad_w)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s InnerProduct W: norm diff %f' % (ok, norm_diff))

    # bias b
    # print('num grad b', num_grad_b)
    # print('grad b', grad_b)
    norm_diff = np.linalg.norm(grad_b - num_grad_b) / np.linalg.norm(grad_b + num_grad_b)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s InnerProduct b: norm diff %f' % (ok, norm_diff))

'''
Tests the analytical gradient computed in the backward pass of the LinearInterpolation layer
against the numerical gradient
'''
def testLinearInterpolation():
    # create input layers
    X = Input(5, 5)
    Y = Input(5, 5)
    L = Input(5, 5)
    x = X.top.value
    y = Y.top.value
    l = L.top.value
    x[:] = np.random.rand(5, 5) - 0.5 # random values between -0.5 and 0.5
    y[:] = np.random.rand(5, 5) - 0.5 # random values between -0.5 and 0.5
    l[:] = np.eye(5)

    # create inner product and mse loss layers
    lerp = LinearInterpolation(X, Y)
    mse_loss = MSELoss(lerp, L)
    mse_loss.loss.grad = 1.0

    lerp.init_params()

    # get analytical gradient
    lerp.forward()
    mse_loss.forward()
    mse_loss.backward()
    lerp.backward()
    grad_x = lerp.bottom1.grad.copy()
    grad_y = lerp.bottom2.grad.copy()
    grad_a = lerp.a_grad

    # compute inputs X numerical gradient
    num_grad_x = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            old_x = x[i][j]

            x[i][j] = old_x + epsilon

            lerp.forward()
            mse_loss.forward()

            num_grad_x[i][j] = mse_loss.loss.value

            x[i][j] = old_x - epsilon

            lerp.forward()
            mse_loss.forward()

            num_grad_x[i][j] -= mse_loss.loss.value
            num_grad_x[i][j] /= 2.0 * epsilon

            x[i][j] = old_x

    # compute inputs Y numerical gradient
    num_grad_y = np.zeros_like(y)

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            old_y = y[i][j]

            y[i][j] = old_y + epsilon

            lerp.forward()
            mse_loss.forward()

            num_grad_y[i][j] = mse_loss.loss.value

            y[i][j] = old_y - epsilon

            lerp.forward()
            mse_loss.forward()

            num_grad_y[i][j] -= mse_loss.loss.value
            num_grad_y[i][j] /= 2.0 * epsilon

            y[i][j] = old_y


    # compute alpha A numerical gradient
    old_a = lerp.a

    lerp.a = old_a + epsilon

    lerp.forward()
    mse_loss.forward()

    num_grad_a = mse_loss.loss.value

    lerp.a = old_a - epsilon

    lerp.forward()
    mse_loss.forward()

    num_grad_a -= mse_loss.loss.value
    num_grad_a /= 2.0 * epsilon

    # inputs X
    # print('num grad x', num_grad_x)
    # print('grad x', grad_x)
    norm_diff = np.linalg.norm(grad_x - num_grad_x) / np.linalg.norm(grad_x + num_grad_x)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s LinearInterpolation X: norm diff %f' % (ok, norm_diff))

    # inputs Y
    # print('num grad y', num_grad_y)
    # print('grad y', grad_y)
    norm_diff = np.linalg.norm(grad_y - num_grad_y) / np.linalg.norm(grad_y + num_grad_y)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s LinearInterpolation Y: norm diff %f' % (ok, norm_diff))

    # alpha A
    # print('num grad a', num_grad_a)
    # print('grad a', grad_a)
    norm_diff = np.linalg.norm(grad_a - num_grad_a) / np.linalg.norm(grad_a + num_grad_a)
    ok = 'GOOD' if norm_diff < 1e-8 else 'BAD'
    print('%s LinearInterpolation A: norm diff %f' % (ok, norm_diff))

testSoftmaxCrossEntropyLoss()
testMSELoss()
testCrossEntropyLoss()
testReLu()
testSin()
testSoftmax()
testInnerProduct()
testLinearInterpolation()
