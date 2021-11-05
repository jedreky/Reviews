"""
This file is used for fast development of simple features.
"""
# Standard library imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Reviews imports
import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config
#######################################################
# Fastdev functions
#######################################################
def test_fun(d):
	d['zaba'] = 'krab'

max_words = 150
quality = 0.75

#analyser.check_score_distribution(max_words, quality)

max_sentences = 30
max_words = 30
emb_dim = 13
GRU_units = 5
Dense_units = [7, 5, 3]

input_shape = (max_sentences, max_words, emb_dim)

r = tf.random.normal( (1, max_sentences, max_words, emb_dim) )

inputs = tf.keras.Input( shape = input_shape )

gru = tf.keras.layers.GRU( GRU_units )

X_list = []

for j in range(max_sentences):
	X_list.append( gru( inputs[ :, j, :, : ] ) )

X = tf.stack(X_list, axis = 1)

req_shape = (-1, max_sentences * GRU_units )

X = tf.reshape( X, shape = req_shape )

for units in Dense_units:
	X = tf.keras.layers.Dense( units, activation = "relu" )(X)

outputs = tf.keras.layers.Dense(1)(X)

model = tf.keras.Model( inputs = inputs, outputs = outputs )
model.summary()

model.predict(r)


rng = np.random.default_rng()

for j in range(0):
	timesteps = 1
	emb_dim = 5
	batch_size = 1

	units = rng.integers(50)
	
	emb_dim = rng.integers(50)
	units = 1
	emb_dim = 1

	input_shape = (timesteps, emb_dim)

	inputs = tf.random.normal( [ batch_size, timesteps, emb_dim ] )

	gru = tf.keras.layers.GRU(units, return_sequences = False)
	output = gru(inputs)
	#print(inputs.shape)
	#print(output.shape)
	#print(gru.get_weights())
	
	w = gru.get_weights()
	
	if w[0].shape != ( emb_dim, 3 * units ):
		print('Error')
	
	if w[1].shape != ( units, 3 * units ):
		print('Error')
	
	if w[2].shape != ( 2, 3 * units ):
		print('Error')
	
	if np.linalg.norm(w[2]) > 0:
		print('Error')

#model = Sequential()
#model.add( GRU(units = 1, input_shape = input_shape ) )
#model.add( Dense(1, activation = None) )
#model.compile( loss = tf.keras.losses.MeanSquaredError(), optimizer = Adam() )

#model.summary()


