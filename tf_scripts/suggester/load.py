import json
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--process", dest="process_string", required=True)
parser.add_argument("-s", "--save", dest="save_file", required=True)
args = parser.parse_args()

# tf.reset_default_graph()
# saver = tf.train.Saver()
# init = tf.global_variables_initializer()

steps = args.process_string
steps_arr = str(steps).split(';')
steps = ' '.join(steps_arr)

save_file = os.getcwd() + '/tf_scripts/suggester/saves/save' + args.save_file

dictionary = dict()
try:
    jsn = open(os.getcwd() + '/tf_scripts/suggester/saves/dict.json', 'r')
    dictionary = dict(json.load(jsn))
except FileNotFoundError:
    print('error opening dict file')
    exit(0)

# print(dictionary)

reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

number_input = 7
number_hidden = 64
vocabulary_size = len(dictionary)

# tf Graph input
# Placeholders for elements that will be always fed in the session
x = tf.placeholder("float", shape=[None, number_input, 1])
y = tf.placeholder("float", shape=[None, vocabulary_size])

w = tf.Variable(tf.random_normal([number_hidden, vocabulary_size]))
b = tf.Variable(tf.random_normal([vocabulary_size]))


# Define the RNN structure
def gru_rnn(current_x, weights, biases):
    # reshape to [1, n_input]
    current_x = tf.reshape(tensor=current_x, shape=[-1, number_input])
    # Generate a n_input-element sequence of inputs (eg. [had] [a] [general] -> [20] [6] [33])
    current_x = tf.split(value=current_x, num_or_size_splits=number_input, axis=1)
    # 2-layer LSTM, each layer has n_hidden units. Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.GRUCell(num_units=number_hidden), rnn.GRUCell(num_units=number_hidden)])
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(cell=rnn_cell, inputs=current_x, dtype=tf.float32)
    # there are n_input outputs but we only want the last output
    return tf.matmul(a=outputs[-1], b=weights) + biases


network = gru_rnn(x, w, b)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()
    # session.run(init)
    saver.restore(session, save_file)
    # print(session)
    # while True:
    #  prompt = "%s words: " % n_input
    #  sentence = input(prompt)
    chain = steps.strip()
    separate_steps = chain.split(' ')
    if len(separate_steps) != number_input:
        print('Length of process is not correct.')
    else:
        try:
            symbols_in_keys = [dictionary[str(separate_steps[i])] for i in range(len(separate_steps))]
            # print(symbols_in_keys)
            keys = np.reshape(np.array(symbols_in_keys), [-1, number_input, 1])
            # print(keys)
            onehot_pred = session.run(network, feed_dict={x: keys})
            # print(onehot_pred)
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            # print(onehot_pred_index)
            chain = "%s %s" % (chain, reverse_dictionary[onehot_pred_index])
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
            print(chain)
        except:
            print('Unknown steps.')
