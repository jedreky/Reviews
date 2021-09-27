"""
This files explores a family of models by optimising them for a specified amount of time.
We assume that the training data has already been generated and stored in an .npz file.
"""

import reviewanalyser.analyser as analyser
import reviewanalyser.auxiliary_functions as aux
import reviewanalyser.config as config

test_mode = True
# The maximal length of a review
max_words = 150

# The following lists represent various options explored in the full mode
learning_rates = (0.0005, 0.001, 0.002)
layers = ('GRU', 'LSTM')
units_options = (32, 64, 128)
n = len( config.emb_dims ) * len( learning_rates ) * len( layers ) * len( units_options )

training_time_test_mode = 1/300
training_time_full_mode = 1/600

print('\nIn test mode we optimise a single model.')
print('In full mode we optimise {} distinct models.\n'.format(n))

if test_mode:
	model_name = 'test_model163'
	input_shape = ( max_words, config.emb_dims[0] )
	params = analyser.generate_params()
	hist = analyser.explore_model( model_name, input_shape, params, training_time_test_mode )
else:
	for emb_dim in config.emb_dims:
		input_shape = ( max_words, emb_dim )

		for learning_rate in learning_rates:
			for layer in layers:
				for units in units_options:
					client = aux.get_client()
					coll = client['ReviewAnalyser']['results']
					count = coll.count_documents({}) + 1
					model_name = 'model{}'.format(count)
					params = analyser.generate_params(learning_rate, layer, units)
					analyser.explore_model( model_name, input_shape, params, training_time_full_mode )
