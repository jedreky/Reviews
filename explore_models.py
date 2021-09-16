import reviewanalyser.analyser as analyser
import reviewanalyser.auxiliary_functions as aux
import reviewanalyser.config as config

test_mode = True
max_words = 150

learning_rates = (0.0005, 0.001, 0.002)
layers = ('GRU', 'LSTM')
units_options = (32, 64, 128)
training_time = 1/600

print('In test mode you optimise a single model.')
n = len( config.emb_dims ) * len( learning_rates ) * len( layers ) * len( units_options )
print('In full mode you optimise {} distinct models.'.format(n))

if test_mode:
	model_name = 'test_model171'
	input_shape = ( max_words, config.emb_dims[0] )
	params = analyser.generate_params()
	analyser.test_model( model_name, input_shape, params, 1/60 )
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
					analyser.test_model( model_name, input_shape, params, training_time )
