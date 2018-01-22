import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
import os

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
import cnn
import network_initializer
import q_loss

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(None, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(None, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

Q_s = cnn.get_network_for_input_raw(x)
Q_s_n = cnn.get_network_for_input_raw(xn)

loss = q_loss.Q_loss(Q_s, u, Q_s_n, ustar, r, term)

optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())

saver = tf.train.Saver()

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it

sess.run(tf.global_variables_initializer())

# steps = 1 * 10**6
steps = 1000
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)


if os.path.exists("./model") is True:
    saver.restore(sess, "./model/")


for step in np.arange(steps):
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    flattened_state_with_history = np.float32(state_with_history.reshape(1, opt.hist_len * opt.state_siz))

    action_tf = tf.argmax(input=Q_s, axis=1)
    action = sess.run(
        action_tf,
        feed_dict = {x : flattened_state_with_history}
    )

    action = action[0]

    print("Action: {}".format(action))

    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    epi_step += 1
    print("Step: {} - Epi step/early_stop: {}/{}".format(step, epi_step, opt.early_stop))
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state

    # TODO every once in a while you should test your agent here so that you can track its performance

    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()


save_path = saver.save(sess, "./model/")

# 2. perform a final test of your model and save it
# TODO

