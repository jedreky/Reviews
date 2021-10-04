"""
This files explores a family of models by optimising them for a specified amount of time.
We assume that the training data has already been generated and stored in an .npz file.
"""

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

mode = 'partial'

# The following lists represent various options explored in the full mode
learning_rates = (0.00025, 0.0005, 0.001, 0.002, 0.004)
layers = ('GRU', 'LSTM')
units_options = (32, 64, 128)
n = len( config.emb_dims ) * len( learning_rates ) * len( layers ) * len( units_options )

training_time_test_mode = 1/60
training_time_partial_mode = 4
training_time_full_mode = 1/60

print('\nIn test mode we optimise a single model.')
print('In full mode we optimise {} distinct models.\n'.format(n))

if mode == 'test':
	input_shape = ( config.max_words, config.emb_dims[0] )
	params = analyser.generate_params( predictor = 'numerical' )
	results = analyser.explore_model( mode, input_shape, params, training_time_test_mode )
	
elif mode == 'partial':
	input_shape = ( config.max_words, config.emb_dims[0] )
	layer = layers[0]
	units = units_options[0]

	for learning_rate in learning_rates:
		params = analyser.generate_params(learning_rate, layer, units)
		results = analyser.explore_model( mode, input_shape, params, training_time_partial_mode )

elif mode == 'full':
	for emb_dim in config.emb_dims:
		input_shape = ( config.max_words, emb_dim )

		for learning_rate in learning_rates:
			for layer in layers:
				for units in units_options:
					params = analyser.generate_params(learning_rate, layer, units)
					results = analyser.explore_model( mode, input_shape, params, training_time_full_mode )
