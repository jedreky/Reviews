"""
This files explores a family of models by optimising them for a specified amount of time.
We assume that the training data has already been generated and stored in an .npz file.
"""

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

batch_name = '3_emb_dims'

layers = ('GRU', 'LSTM')
units_options = (32, 64, 128)

training_time_test_mode = 1/600
training_time = 1/600

if batch_name == 'test':
	input_shape = ( config.max_words, config.emb_dims[0] )
	params = analyser.generate_params( predictor = 'numerical' )
	results = analyser.explore_model( batch_name, input_shape, params, training_time_test_mode )

elif batch_name == '1_learning rate':
	learning_rates = (0.00025, 0.0005, 0.001, 0.002, 0.004)
	input_shape = ( config.max_words, config.emb_dims[0] )
	layer = layers[0]
	units = units_options[0]

	for learning_rate in learning_rates:
		params = analyser.generate_params(learning_rate, layer, units)
		results = analyser.explore_model( batch_name, input_shape, params, training_time )

elif batch_name == '2_layers_units':
	learning_rate = 0.0015
	input_shape = ( config.max_words, config.emb_dims[0] )

	for layer in layers:
		for units in units_options:
			params = analyser.generate_params(learning_rate, layer, units)
			results = analyser.explore_model( batch_name, input_shape, params, training_time )

elif batch_name == '3_emb_dims':
	learning_rate = 0.0015
	params = analyser.generate_params(learning_rate, layers[0], units_options[0])
	
	input_shape = ( config.max_words, config.emb_dims[0] )
	results = analyser.explore_model( batch_name, input_shape, params, 7, 'results/2_layers_units/1_final.h5' )
	
	input_shape = ( config.max_words, config.emb_dims[1] )
	results = analyser.explore_model( batch_name, input_shape, params, 11 )

else:
	print('Unknown batch name.')

aux.send_email('Running explore_models.py terminated successfully.')
