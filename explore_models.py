"""
This files explores a family of models by optimising them for a specified amount of time.
We assume that the training data has already been generated and stored in an .npz file.
"""

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

filename = 'data'
batch_name = '4_padding'

layers = ('GRU', 'LSTM')
units_options = (32, 64, 128)

training_time = 1/1000

client = aux.get_client()

if batch_name == 'test':
	params = analyser.generate_params()
	params['data_file'] = 'input_data/data{}d.npz'.format( params['input_shape'][1] )
#	params['parent_id'] = ('test', 4)
	results = analyser.setup_and_train_model(client, batch_name, params, 1/500 )

elif batch_name == '1_learning rate':
	learning_rates = (0.00025, 0.0005, 0.001, 0.002, 0.004)
	input_shape = ( config.max_words, config.emb_dims[0] )
	layer = layers[0]
	units = units_options[0]

	for learning_rate in learning_rates:
		params = analyser.generate_params(learning_rate, layer, units)
		results = analyser.setup_and_train_model(client, batch_name, input_shape, params, training_time )

elif batch_name == '2_layers_units':
	learning_rate = 0.0015
	input_shape = ( config.max_words, config.emb_dims[0] )

	for layer in layers:
		for units in units_options:
			params = analyser.generate_params(learning_rate, layer, units)
			results = analyser.setup_and_train_model(client, batch_name, input_shape, params, training_time )

elif batch_name == '3_emb_dims':
	learning_rate = 0.0015
	params = analyser.generate_params(learning_rate, layers[0], units_options[0])
	
	input_shape = ( config.max_words, config.emb_dims[0] )
	results = analyser.setup_and_train_model(client, batch_name, input_shape, params, 7, 'results/2_layers_units/1_final.h5' )
	
	input_shape = ( config.max_words, config.emb_dims[1] )
	results = analyser.setup_and_train_model(client, batch_name, input_shape, params, 11 )

elif batch_name == '4_padding':
	learning_rate = 0.0015
	
	for padding in ('pre', 'post'):
		params = analyser.generate_params(learning_rate, layers[0], units_options[0])
		params['data_file'] = 'input_data/{}{}d-{}.npz'.format( filename, params['input_shape'][1], padding )
		analyser.setup_and_train_model(client, batch_name, params, training_time )

else:
	print('Unknown batch name.')

client.close()
aux.send_email('Running explore_models.py terminated successfully.')
