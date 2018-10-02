import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tflearn.datasets.mnist as mnist

X, Y, testX, testY = mnist.load_data(one_hot=True)
X = np.reshape(X, (-1, 28, 1, 28))
testX = np.reshape(testX, (-1, 28, 1, 28))

def convlstmnet():
    image = tflearn.input_data(shape=[None, 28, 1, 28])
    rnn_cell = tf.contrib.rnn.Conv1DLSTMCell(input_shape=[1,28],kernel_shape=[3],output_channels=32)
    rnn_cell2 = tf.contrib.rnn.Conv1DLSTMCell(input_shape=[1,28],kernel_shape=[3],output_channels=16)
    lstm_multi = tf.nn.rnn_cell.MultiRNNCell([rnn_cell, rnn_cell2], state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(lstm_multi,image,dtype=tf.float32)
    return outputs[:,-1,:]

net = convlstmnet()
output = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(output, optimizer='adam',
                         loss='categorical_crossentropy', name="output")
model = tflearn.DNN(net, tensorboard_verbose=1)
model.fit(X, Y, n_epoch=10, validation_set=1e-3, show_metric=True,
          snapshot_step=100)
