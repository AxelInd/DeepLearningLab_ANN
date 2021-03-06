import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
import tensorflow as tf
import cnn

def main(unused_argv):
    classifier = cnn.get_estimator()

    # 0. initialization
    opt = Options()
    sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

    # TODO: load your agent
    # Hint: If using standard tensorflow api it helps to write your own model.py  
    # file with the network configuration, including a function model.load().
    # You can use saver = tf.train.Saver() and saver.restore(sess, filename_cpkt)

    agent =None

    # 1. control loop
    if opt.disp_on:
        win_all = None
        win_pob = None
    epi_step = 0    # #steps in current episode
    nepisodes = 0   # total #episodes executed
    nepisodes_solved = 0
    action = 0     # action to take given by the network

    # start a new game
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    for step in range(opt.eval_steps):

        # check if episode ended
        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            nepisodes += 1
            if state.terminal:
                nepisodes_solved += 1
            # start a new game
            state = sim.newGame(opt.tgt_y, opt.tgt_x)
        else:
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # TODO: here you would let your agent take its action
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Hint: get the image using rgb2gray(state.pob), append latest image to a history 
            # this just gets a random action
            # action = randrange(opt.act_num)
            
            image = rgb2gray(state.pob)
            reshaped_image = np.zeros(opt.state_siz)
            flattened_image = image.flatten()[0:opt.state_siz]
            reshaped_image[0:flattened_image.shape[0]] = flattened_image
            reshaped_image = reshaped_image.reshape(1, opt.state_siz)

            if epi_step == 0:
                images_history = np.zeros([opt.hist_len, opt.state_siz])
                for j in range(opt.hist_len):
                    images_history[j] = reshaped_image
            else:
                images_history = np.append(images_history, reshaped_image, 0)
                if images_history.shape[0] > opt.hist_len:
                    images_history = np.delete(images_history, 0, 0)

            images_history_flatten = np.float32(
                images_history.reshape(1, opt.state_siz * opt.hist_len)
            )

            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": images_history_flatten},
                num_epochs=1,
                shuffle=False
            )

            predictions = classifier.predict(input_fn=predict_input_fn)
            for i, p in enumerate(predictions):
                action_predicted = p["classes"]
                break

            state = sim.step(action_predicted)

            epi_step += 1

        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            nepisodes += 1
            if state.terminal:
                nepisodes_solved += 1
            # start a new game
            state = sim.newGame(opt.tgt_y, opt.tgt_x)

        if step % opt.prog_freq == 0:
            print(step)

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

    # 2. calculate statistics
    print("Result for history length: {}".format(opt.hist_len))
    print(float(nepisodes_solved) / float(nepisodes))
    # 3. TODO perhaps  do some additional analysis

if __name__ == "__main__":
  tf.app.run()