#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from layers import *
from nn import *

batch_size = 100

# NN architecture
nn = NeuralNet()
i = nn.add('data', Input(batch_size, 784))
l = nn.add('labels', Input(batch_size, 10))
'''
fc1 = nn.add('fc1', InnerProduct(i, 10))
relu1 = nn.add('relu1', ReLu(fc1))
fc2 = nn.add('fc2', InnerProduct(relu1, 10))
relu2 = nn.add('relu2', ReLu(fc2))
fc3 = nn.add('fc3', InnerProduct(relu2, 10))
relu3 = nn.add('relu3', ReLu(fc3))
fc4 = nn.add('fc4', InnerProduct(relu3, 10))
relu4 = nn.add('relu4', ReLu(fc4))
'''
fc5 = nn.add('fc5', InnerProduct(i, 10))
softmax = nn.add('softmax', Softmax(fc5))
loss = nn.add('loss', CrossEntropyLoss(softmax, l))

step_size = 0.005

nn.init_params()

# plot setup
plt.ion()

train_fig = plt.figure()
t_ax = train_fig.add_subplot(111)
acc_ax = t_ax.twinx()

t_loss_line, = t_ax.plot([], [])
t_ax.set_xlabel('Iterations')
t_ax.set_ylabel('train loss')

acc_line, = acc_ax.plot([], [])
acc_ax.set_ylabel('validation accuracy')

train_losses = []
val_accuracy = []
val_iter = []

# training
epochs = 5
train_samples = 55000
validation_samples = 5000
num_batches = train_samples / batch_size
test_interval = 50

for epoch in range(epochs):
    for batch in range(num_batches):
        if batch % 10 == 0:
            print('Batch %d/%d...' % (batch, num_batches))

        # load batch
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        i.top.value[...] = batch_x
        l.top.value[...] = batch_y

        nn.forward()

        nn.reset_gradients()

        # minimise loss
        loss.top.grad = -1.0

        nn.backward()
        nn.update_params(step_size)

        # validation
        if batch % test_interval == 0:
            correct_samples = 0

            for b in range(validation_samples / batch_size):
                test_batch_x, test_batch_y = mnist.validation.next_batch(batch_size)
                i.top.value[...] = test_batch_x
                l.top.value[...] = test_batch_y

                nn.forward()

                correct_samples += np.sum(np.argmax(softmax.top.value, axis = 1) == np.argmax(test_batch_y, axis = 1))

            val_acc = float(correct_samples) / validation_samples * 100
            print('Validation accuracy %g' % val_acc)

            val_accuracy.append(val_acc)
            val_iter.append(epoch * num_batches + batch)

            acc_line.set_xdata(val_iter)
            acc_line.set_ydata(val_accuracy)
            acc_ax.relim()
            acc_ax.autoscale_view()
        
        '''
        # learning rate decay (horrible)
        if len(val_losses) > 100 and batch % 100 == 0 and abs(val_losses[-100] - val_losses[-1]) < 0.0001:
            step_size /= 10
            print('new step size', step_size)
            t_ax.axvline(batch)
        '''

        # plot train loss over iterations
        train_losses.append(loss.top.value)
        t_loss_line.set_xdata(np.arange(len(train_losses)))
        t_loss_line.set_ydata(train_losses)
        t_ax.relim()
        t_ax.autoscale_view()
        train_fig.canvas.draw()

        plt.pause(0.05)

    # plot epoch line
    t_ax.axvline((epoch + 1) * num_batches, color='r')
    train_fig.canvas.draw()
    plt.pause(0.05)

print('Training finished')
print('Testing accuracy...')

# measure test accuracy
test_samples = 10000
correct_samples = 0
for batch in range(test_samples / batch_size):
    # load batch
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    i.top.value[...] = batch_x
    l.top.value[...] = batch_y

    nn.forward()

    correct_samples += np.sum(np.argmax(softmax.top.value, axis = 1) == np.argmax(batch_y, axis = 1))

print('Correct: %d/%d' % (correct_samples, test_samples))
print('Accuracy: %g' % (float(correct_samples) / test_samples * 100))

# simple test
i.top.value[0, ...] = mnist.test.images[0]

nn.forward()

print('Final %r' % softmax.top.value[0])
print('Prediction %d' % np.argmax(softmax.top.value[0]))
print('label', mnist.test.labels[0])
plt.figure()
plt.imshow(i.top.value[0].reshape(28, 28), cmap='gray')

# dont close the window
while True:
    plt.pause(0.05)

