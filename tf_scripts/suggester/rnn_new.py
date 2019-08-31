""" A GRU-based recurrent neural network implementation using TensorFlow.
Predicts the next architectural design step for a given process chain.
Based on the RNN from https://github.com/roatienza/Deep-Learning-Experiments """

from __future__ import print_function

import collections
import json
import random
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="training_file")
parser.add_argument("-s", "--save", dest="save_file")
args = parser.parse_args()

# File containing steps for training
training_file = os.getcwd() + '/tf_scripts/suggester/cases_csv/' + args.training_file + '.txt'

# File for saving of the session
save_file = os.getcwd() + '/tf_scripts/suggester/saves/save' + args.save_file


def create_dataset(actions):
    # print(actions)
    # smth like ['a0L' 'a1P' 'a0R' ... 'r1C' 't1S' 'r1K']
    count = collections.Counter(actions).most_common()
    # print(count)
    # smth like [('a0S', 50), ('a1B', 46), ('a1L', 45), ('a1R', 45), ('a0R', 44), ...]
    current_dictionary = dict()
    for word, _ in count:
        current_dictionary[word] = len(current_dictionary)
    with open(os.getcwd() + '/tf_scripts/suggester/saves/dict.json', 'w') as d:
        d.write(json.dumps(current_dictionary, indent=2))
    # print(current_dictionary)
    # smth like [('a0S', 0), ('a1B', 1), ('a1L', 3), ('a1R', 4), ...]
    reverse_current_dictionary = dict(zip(current_dictionary.values(), current_dictionary.keys()))
    # print(reverse_current_dictionary)
    # smth like {0: 'a0S', 1: 'a1B', 2: 'a1R', 3: 'a1L', 4: 'a0R', ...}
    return current_dictionary, reverse_current_dictionary


with open(training_file) as f:
    data = f.readlines()
data = [c.strip() for c in data]
data = [word for i in range(len(data)) for word in data[i].split()]
training_data = np.array(data)

dictionary, reverse_dictionary = create_dataset(training_data)
vocabulary_size = len(dictionary)
# e.g. 72

# Parameters
learning_rate = 0.01
training_iterations = 3000
display_step = 1000
number_input = 7
# number of units in RNN cell
number_hidden = 64

# tf Graph input
# Placeholders for elements that will be always fed in the session
x = tf.placeholder("float", shape=[None, number_input, 1])
y = tf.placeholder("float", shape=[None, vocabulary_size])

# RNN output node weights and biases
# tf random normal = Outputs random values from a normal distribution for a given shape, e.g.: [256, 72]
w = tf.Variable(tf.random_normal([number_hidden, vocabulary_size]))
b = tf.Variable(tf.random_normal([vocabulary_size]))


# Define the RNN structure
def gru_rnn(current_x, weights, biases):
    # reshape to [1, n_input]
    current_x = tf.reshape(tensor=current_x, shape=[-1, number_input])
    # Generate a n_input-element sequence of inputs (eg. [had] [a] [general] -> [20] [6] [33])
    current_x = tf.split(value=current_x, num_or_size_splits=number_input, axis=1)
    # 2-layer GRU, each layer has n_hidden units. Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.GRUCell(num_units=number_hidden), rnn.GRUCell(num_units=number_hidden)])
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(cell=rnn_cell, inputs=current_x, dtype=tf.float32)
    # there are n_input outputs but we only want the last output
    return tf.matmul(a=outputs[-1], b=weights) + biases


network = gru_rnn(x, w, b)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_predictions = tf.equal(tf.argmax(network, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initializing the variables
initializer = tf.global_variables_initializer()

with tf.Session() as session:
    saver = tf.train.Saver()
    session.run(initializer)
    step = 0
    offset = random.randint(0, number_input + 1)
    end_offset = number_input + 1
    acc_total = 0
    loss_total = 0

    while step < training_iterations:
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, number_input + 1)
        symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + number_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, number_input, 1])

        symbols_out_onehot = np.zeros([vocabulary_size], dtype=float)
        # e.g. [0. 0. 0. 0. ... 0.]
        symbols_out_onehot[dictionary[str(training_data[offset + number_input])]] = 1.0
        # e.g. [0. 0. 1. 0. 0. ... 0]
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pred = session.run(fetches=[optimizer, accuracy, cost, network],
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Loss= " +
                  "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " +
                  "{:.2f}%".format(100 * acc_total / display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + number_input)]
            symbols_out = training_data[offset + number_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
        step += 1
        offset += (number_input + 1)
    print("Optimization Finished!")
    saver.save(session, save_file)
