"""
This file calls the create_model function with various parameters and performs optimisation on random data.
The goal is to check that the models are created successfully and with correct input/output dimensions.
"""

import numpy as np
import tensorflow as tf

import reviews.analyser as analyser

# number of random tests to perform for which case
N_test = 5
# upper limit on the number of dense layers
m_Dense_layers = 4
# upper limit on the remaining random parameters
m = 30

rng = np.random.default_rng()

for sentence_based in (True, False):
	for RNN_type in ('GRU', 'LSTM'):
		for predictor in ('numerical', 'categorical'):
			for j in range(N_test):
				RNN_units = rng.integers(m)
				max_sentences = rng.integers(m)
				max_words = rng.integers(m)
				emb_dim = rng.integers(m)
				Dense_units = []
				
				for k in range( rng.integers(low = 1, high = m_Dense_layers) ):
					Dense_units.append( rng.integers(low = 1, high = m) )
			
				params = analyser.generate_params( sentence_based = sentence_based, RNN_type = RNN_type, RNN_units = RNN_units, Dense_units = Dense_units, predictor = predictor, max_sentences = max_sentences, max_words = max_words, emb_dim = emb_dim )
				model = analyser.create_model(params)
				model.summary()
				
				batch_size = (10, )
				X = tf.random.normal( batch_size + params['input_shape'] )
				Y = tf.random.uniform( shape = batch_size, minval = 0, maxval = 9, dtype=tf.int64 )
				model.fit( X, Y, epochs = 5 )
