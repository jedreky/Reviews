"""
This test calls the create_model function with various parameters and performs a couple of optimisation steps on random data.
The goal is to check that the models are created successfully and with correct input/output dimensions.
"""
# Standard library imports
import numpy as np
import tensorflow as tf

# Reviews imports
import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
#######################################################
# Test various types of models
#######################################################
def test_random_models():
	# number of random tests to perform for each case
	N_tests_per_case = 5
	# number of epochs to train for
	N_epochs = 5
	# batch size to use when generating random data
	batch_size = (10, )

	# upper limit on the number of dense layers
	m_Dense_layers = 4
	# upper limit on the remaining random parameters
	m = 30

	N_tests = 8 * N_tests_per_case

	# initialise a random number generator
	rng = np.random.default_rng()

	# initialise a counter
	ct = 0

	# iterate over different types of models
	for sentence_based in (True, False):
		for RNN_type in ('GRU', 'LSTM'):
			for predictor in ('numerical', 'categorical'):
				for j in range(N_tests_per_case):
					# choose random parameters
					RNN_units = rng.integers(low = 1, high = m)
					max_words = rng.integers(low = 1, high = m)
					max_sentences = rng.integers(low = 1, high = m)
					max_words_per_sentence = rng.integers(low = 1, high = m)
					emb_dim = rng.integers(low = 1, high = m)
					Dense_units = []

					# choose a structure of the Dense layers
					for k in range( rng.integers(low = 1, high = m_Dense_layers) ):
						Dense_units.append( rng.integers(low = 1, high = m) )

					# generate a dictionary containing the parameters and print it
					params = analyser.generate_params( sentence_based = sentence_based, RNN_type = RNN_type, RNN_units = RNN_units, Dense_units = Dense_units, predictor = predictor, max_words = max_words, max_sentences = max_sentences, max_words_per_sentence = max_words_per_sentence, emb_dim = emb_dim )
					ct += 1
					aux.log('Running test {}/{}'.format(ct, N_tests))
					aux.log('Model parameters: {}'.format(params))
					# create the model
					model = analyser.create_model(params)
					# print the summary
					model.summary()

					# generate random data and perform some training
					X = tf.random.normal( batch_size + params['input_shape'] )
					Y = tf.random.uniform( shape = batch_size, minval = 0, maxval = 9, dtype = tf.int64 )
					model.fit( X, Y, epochs = N_epochs )
					aux.log('Model generated and trained successfully.\n')

if __name__ == '__main__':
	test_random_models()
