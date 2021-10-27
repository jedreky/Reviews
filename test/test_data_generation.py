"""
This test generates input data and then uses it to train a couple of models.
The goal is to check that the format of the input data matches what we expect.
"""

import numpy as np
import tensorflow as tf

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

def test_data_generation():
	N_reviews = 5
	filename = 'data'
	criteria = {'max_words': 50, 'max_sentences': 10, 'max_words_per_sentence': 20}
	
	# generate input data
	aux.log('Generate input data from the reviews stored in the database.')
	client = aux.get_client()
	analyser.generate_input_data(client, filename, N_reviews, criteria)
	client.close()

	N_epochs = 5

	aux.log('For each data file construct a model and train it.')
	# for each data file generate a model and train it on both train and test data
	for emb_dim in config.emb_dims:
		for sentence_based in (True, False):
			for padding in ('pre', 'post'):
				f = 'input_data/{}-{}-{}d-{}.npz'.format( filename, sentence_based, emb_dim, padding )
				X_train, X_test, Y_train, Y_test = analyser.load_data(f)
				
				params = analyser.generate_params( sentence_based = sentence_based, max_words = criteria['max_words'], max_sentences = criteria['max_sentences'], max_words_per_sentence = criteria['max_words_per_sentence'], emb_dim = emb_dim )
				aux.log('Model parameters: {}'.format(params))
				# create the model
				model = analyser.create_model(params)
				# print the summary
				model.summary()

				# perform some training
				model.fit( X_train, Y_train, epochs = N_epochs )
				model.fit( X_test, Y_test, epochs = N_epochs )

if __name__ == '__main__':
	test_data_generation()
