"""
This file quickly builds a simple model, so that we can test it.
"""

import reviews.analyser as analyser
import reviews.config as config

max_words = 150
input_shape = ( max_words, config.emb_dims[0] )
data_file = 'data/data{}d.npz'.format( input_shape[1] )
X_train, X_test, Y_train, Y_test = analyser.load_data(data_file)

params = analyser.generate_params( predictor = 'categorical' )
model = analyser.create_model(input_shape, params)
model.summary()
model.fit( X_train, Y_train, epochs = 5 )
