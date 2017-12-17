import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
import cnn

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
def main(unused_argv):
	opt = Options()
	sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
	trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
	                             opt.minibatch_size, opt.valid_size,
	                             opt.states_fil, opt.labels_fil)

	# 1. train
	######################################
	# TODO implement your training here!
	# you can get the full data from the transition table like this:
	#
	# # both train_data and valid_data contain tupes of images and labels
	train_data = trans.get_train()
	valid_data = trans.get_valid()

	samples_train_data = np.float32(train_data[0])
	labels_train_data = np.float32(train_data[1])
	unhotted_labels_train_data = unhot(labels_train_data)

	samples_valid_data = np.float32(valid_data[0])
	labels_valid_data = np.float32(valid_data[1])
	unhotted_labels_valid_data = unhot(labels_valid_data)

	print("Shape of samples_train_data {}".format(samples_train_data.shape))
	print("Shape of labels_train_data {}".format(labels_train_data.shape))
	print("Shape of unhotted_labels_train_data {}".format(unhotted_labels_train_data.shape))

	classifier = cnn.get_estimator()

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": samples_train_data},
	    y=unhotted_labels_train_data,
	    batch_size=100,
	    num_epochs=None,
	    shuffle=True)

	classifier.train(
	    input_fn=train_input_fn,
	    steps=1000
	)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": samples_valid_data},
		y=unhotted_labels_valid_data,
		num_epochs=1,
		shuffle=False
	)

	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

	# 
	# alternatively you can get one random mini batch line this
	#
	# for i in range(number_of_batches):
	#     x, y = trans.sample_minibatch()
	# Hint: to ease loading your model later create a model.py file
	# where you define your network configuration
	######################################

	# 2. save your trained model

def unhot(one_hot_labels):
    """ Invert a one hot encoding, creating a flat vector """
    return np.argmax(one_hot_labels, axis=-1)

if __name__ == "__main__":
  tf.app.run()