from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import Options, rgb2gray
import q_loss

tf.logging.set_verbosity(tf.logging.INFO)

def get_network_for_input_raw(state_batch):
  opt = Options()
  depth = opt.hist_len
  input_size = np.int32(np.sqrt(opt.state_siz))
  pool_size = 2
  stride_size = 2

  with tf.variable_scope("DQN", reuse=tf.AUTO_REUSE):
    # print("Input size: {}".format(input_size))

    # (batch, depth, height, width, channels)
    input_layer = tf.reshape(state_batch, [-1, depth, input_size, input_size, 1])

    # print("Shape of state_batch: {}".format(input_layer.shape))

    # Convolutional Layer #1
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # print("Shape of conv1: {}".format(conv1))

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling3d(
      inputs=conv1,
      pool_size=[1, pool_size, pool_size],
      strides=(1, stride_size, stride_size)
    )

    # print("Shape of pool1: {}".format(pool1))

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # print("Shape of conv2: {}".format(conv2))

    pool2 = tf.layers.max_pooling3d(
      inputs=conv2,
      pool_size=[depth, 2, 2],
      strides=(depth, 2, 2)
    )

    # print("Shape of pool2: {}".format(pool2))

    size_input_after_two_pools = input_size // (2 * pool_size)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, size_input_after_two_pools * size_input_after_two_pools * 32])

    # print("Shape of pool2_flat: {}".format(pool2_flat))

    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    if (state_batch.shape[0] > 1):
      trainingMode = True
    else:
      trainingMode = False

    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4,
      training=trainingMode
    )

    # print("Shape of dense: {}".format(dense))

    # Logits Layer
    q_s = tf.layers.dense(inputs=dropout, units=5)
    # print("Shape of q_s: {}".format(q_s))

    return q_s
