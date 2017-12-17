from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import Options, rgb2gray
import cnn

def get_estimator():
  opt = Options()

  model_dir = "./cnn_model/hist_len_{}_pob_siz_{}_cub_siz_{}".format(opt.hist_len, opt.pob_siz, opt.cub_siz)

  return tf.estimator.Estimator(
    model_fn=cnn.cnn_model_fn, model_dir=model_dir
  )


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # HARDCODED!!!!

  opt = Options()
  depth = opt.hist_len
  input_size = np.int32(np.sqrt(opt.state_siz))
  pool_size = 2
  stride_size = 2

  # print("Input size: {}".format(input_size))

  # (batch, depth, height, width, channels)
  input_layer = tf.reshape(features["x"], [-1, depth, input_size, input_size, 1])

  # print("Shape of input: {}".format(input_layer.shape))

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
      filters=64,
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
  pool2_flat = tf.reshape(pool2, [-1, size_input_after_two_pools * size_input_after_two_pools * 64])

  # print("Shape of pool2_flat: {}".format(pool2_flat))

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # print("Shape of dense: {}".format(dense))

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # print("Shape of dropout: {}".format(dropout))

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=5)

  # print("Shape of logits: {}".format(logits))

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)

  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)