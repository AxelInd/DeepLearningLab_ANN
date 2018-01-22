import cnn
import numpy as np
from utils import Options
import tensorflow as tf

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# setup placeholders for states (x) actions (u) and rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

# get the output from your network
Q = my_network_forward_pass(x)
Qn =  my_network_forward_pass(xn)

# calculate the loss
loss = Q_loss(Q, u, Qn, ustar, r, term)

# setup an optimizer in tensorflow to minimize the loss
"""

def init_raw():
	x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
	u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
	ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
	xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
	r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
	term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

	Q_s = cnn.get_network_for_input_raw(x)
	Q_s_n = cnn.get_network_for_input_raw(xn)

	loss = Q_loss
