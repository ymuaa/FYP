import tensorflow as tf
import numpy as np

# basic
cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128)
print(cell.state_size)  # 128

basic_inputs = tf.placeholder(np.float32, shape=(32, 100))
basic_h0 = cell.zero_state(32, np.float32)

# h1 = f(U * x_1 + W * h_0 + b)
basic_outputs, basic_h1 = cell.call(basic_inputs, basic_h0)

print(basic_h1.shape) # (32, 128)

tf.reset_default_graph()    
# https://stackoverflow.com/questions/47296969/valueerror-variable-rnn-basic-rnn-cell-kernel-already-exists-disallowed-did-y

# lstm
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = lstm_cell.zero_state(32, np.float32)
outputs, h1 = lstm_cell.call(inputs, h0)

# LSTM has 2 hidden status h and c:
print(h1.h)     # Tensor("mul_2:0", shape=(32, 128), dtype=float32)
print(h1.c)     # Tensor("add_1:0", shape=(32, 128), dtype=float32)
