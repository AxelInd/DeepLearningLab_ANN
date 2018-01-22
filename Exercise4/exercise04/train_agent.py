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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())

saver = tf.train.Saver()

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it

sess.run(tf.global_variables_initializer())

# steps = 1 * 10**6
steps = 5001
epi_step = 0
nepisodes = 0
nepisodes_solved = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)


if os.path.exists("./model") is True:
    saver.restore(sess, "./model/")

f = open("output_training.log", 'w')

for step in np.arange(steps):
    if state.terminal or epi_step >= opt.early_stop:
        if state.terminal:
            nepisodes_solved += 1

        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history

        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would let your agent take its action
    #       remember
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # this just gets a random action

    if np.random.uniform() * 100 > 20:
        flattened_state_with_history = np.float32(state_with_history.reshape(1, opt.hist_len * opt.state_siz))

        action_tf = tf.argmax(input=Q_s, axis=1)
        action = sess.run(
            action_tf,
            feed_dict = {x : flattened_state_with_history}
        )

        action = action[0]
    else:
        action = randrange(opt.act_num)

    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    epi_step += 1
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would train your agent
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

    action_batch_next_tf = tf.argmax(input=Q_s_n, axis=1)
    action_batch_next_not_onehot = sess.run(
        action_batch_next_tf,
        feed_dict = {xn : next_state_batch}
    )

    action_batch_next = np.zeros((opt.minibatch_size, opt.act_num))

    for i in range(len(action_batch_next_not_onehot)):
        action_batch_next[i] = trans.one_hot_action(action_batch_next_not_onehot[i])

    sess.run(
        train_op,
        feed_dict = {
            x: state_batch,
            u: action_batch,
            ustar: action_batch_next,
            xn: next_state_batch,
            r: reward_batch,
            term: terminal_batch
        }
    )
    if step%100 ==0:
        print("Step",step)
    if step % 500 == 0:
        loss_value = sess.run(
            loss,
            feed_dict = 
                {
                    x : state_batch,
                    u : action_batch,
                    ustar : action_batch_next,
                    xn : next_state_batch,
                    r : reward_batch,
                    term : terminal_batch
                }
        )

        save_path = saver.save(sess, "./model/")

        if nepisodes != 0:
            accuracy = nepisodes_solved / nepisodes
        else:
            accuracy = 0
            
        nepisodes=0
        nepisodes_solved=0
        training_info = "{0},{1:.4f},{2}\n".format(step, loss_value, accuracy)
        print(training_info)
        f.write(training_info)

    # TODO train me here
    # this should proceed as follows:
    # 1) pre-define variables and networks as outlined above
    # 1) here: calculate best action for next_state_batch
    # TODO:
    # action_batch_next = CALCULATE_ME
    # 2) with that action make an update to the q values
    #    as an example this is how you could print the loss 

    
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
f.close()

# 2. perform a final test of your model and save it
# TODO

