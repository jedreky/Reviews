"""
This file contains functions related to analysing the reviews.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn.model_selection
import sklearn.utils
import time

from tensorflow.keras.layers import Dense, GRU, LSTM
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import reviews.auxiliary_functions as aux
import reviews.config as config

#######################################################
# Functions related to data verification and processing
#######################################################
def check_score_distribution():
	"""
	Checks the distribution of scores in the reviews database.
	"""
	coll, client = aux.get_collection('reviews')

	count_by_score = { '$group': { '_id': '$score', 'count': { '$sum': 1 } } }
	pipeline = [ count_by_score ]
	results = coll.aggregate( pipeline )

	for r in results:
		print(r)

	client.close()

def get_input_data(n, max_words, emb_dim, quality):
	"""
	Returns an embedding of reviews from the database that satisfy certain criteria	(maximum number of words and minimum quality).
	To ensure that we are training on a balanced set we choose the same number of reviews (denoted by n) with each grade.
	In the last step the reviews are randomly permuted.
	"""
	coll, client = aux.get_collection('reviews')

	X = np.zeros( [ 10 * n, max_words, emb_dim ] )
	Y = np.zeros( [ 10 * n, 1] )

	with open('glove.6B/glove.6B.{}d.pickle'.format( str(emb_dim) ), 'rb') as pickle_file:
		emb_dict = pickle.load( pickle_file )

	j = 0

	for score in range(1, 11):
		results = coll.find( { 'words': { '$lt': max_words }, 'quality': { '$gt': quality }, 'score': score } )

		k = 0
		for r in results:
			review_emb = convert_review( r['content'], max_words, emb_dict )

			if review_emb is not None:
				X[j + k, :, :] = review_emb
				# subtract 1 to ensure that the scores are in the range [0, 1, ..., 9]
				Y[j + k] = r['score'] - 1
				k += 1

			if k >= n:
				j += k
				k = 0
				break

	client.close()
	
	# truncate the empty rows
	X = X[:j, :, :]
	Y = Y[:j]
	# shuffle the dataset
	X, Y = sklearn.utils.shuffle(X, Y)

	aux.log('Extracting input data for: n = {}, max_words = {}, emb_dim = {}, quality = {}'.format(n, max_words, emb_dim, quality))

	if j == 10 * n:
		aux.log('A sufficient number of reviews found.')
	else:
		aux.log('Insufficient reviews, the resulting matrices are smaller than asked for.')
	
	return X, Y

def save_input_data_to_file(filename, n, max_words, emb_dim, quality):
	"""
	Generates training data for a fixed embedding dimension and saves it to an .npz file.
	"""
	X, Y = get_input_data(n, max_words, emb_dim, quality)
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y, test_size = 0.15 )

	with open('data/{}d.npz'.format( filename ), 'wb') as data_file:
		np.savez(data_file, X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test)

def generate_input_data(filename, n = 15, max_words = 150, quality = 0.5):
	"""
	Generates training data for all valid embedding dimensions and saves them to .npz files.
	"""
	for emb_dim in config.emb_dims:
		save_input_data_to_file( filename + str(emb_dim), n, max_words, emb_dim, quality)


def convert_review(review, length, emb_dict):
	"""
	Converts a review into its embedding.
	"""
	# convert to lower case
	review = review.lower()
	
	# separate symbols
	symbols_to_separate = ( ':' , ',' , '.', '!', '-', '(', ')' )
	
	for symbol in symbols_to_separate:
#		review = review.replace( symbol, ' {} '.format(symbol) )
		review = review.replace( symbol, ' ' )
	
	words = review.split()
	
	review_emb = []
	
	for word in words:
		if word in emb_dict:
			review_emb.append( emb_dict[word] )
#		else:
#			#print('Word "{}" missing'.format(word))

	if len(review_emb) <= length:
		review_emb_pad = np.pad( np.array(review_emb), ( ( length - len(review_emb) , 0), (0, 0) ) )
	else:
		review_emb_pad = None

	return review_emb_pad

def load_data(data_file):
	"""
	Loads data from an .npz file.
	"""
	data = np.load(data_file)
	return data['X_train'], data['X_test'], data['Y_train'], data['Y_test']
#######################################################
# Functions related to ML models
#######################################################
def create_model(input_shape, params):
	"""
	Creates a recurrent neural network according to the specified parameters.
	"""
	model = Sequential()
	if params['layer'] == 'GRU':
		model.add( GRU(units = params['units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout'], input_shape = input_shape ) )
	elif params['layer'] == 'LSTM':
		model.add( LSTM(units = params['units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout'], input_shape = input_shape ) )
	else:
		aux.log('Warning: the layer should be either GRU or LSTM')

	model.add( Dense(10, activation = 'softmax') )
	model.compile( loss = SparseCategoricalCrossentropy(), optimizer = Adam( learning_rate = params['learning_rate'] ), metrics = [SparseCategoricalAccuracy()] )
	return model

def check_accuracy(model, X, Y, err = 1):
	"""
	Returns the fraction of examples for which the returned score is close up to some fixed error to the true score.
	"""
	count = 0
	for j in range( X.shape[0] ):
		Y_pred = np.argmax( model.predict( X[j].reshape( [1, X.shape[1], X.shape[2]] ) ) )
		if np.abs( Y_pred - Y[j] ) <= err:
			count += 1

	return float(count / X.shape[0])

def predict_rating(model, review, length, emb_dim):
	"""
	Given a model and a review returns the prediction.
	"""
	with open('glove.6B/glove.6B.{}d.pickle'.format( str(emb_dim) ), 'rb') as pickle_file:
		emb_dict = pickle.load( pickle_file )

	x = convert_review( review, length, emb_dict )
	y = model.predict( x.reshape( [1, x.shape[0], x.shape[1] ] ) )
	print( np.round(y, 3) )

def train_and_evaluate_model(model, data_file, time_in_hrs):
	"""
	Trains the given model for a required amount of time and evaluates its performance on the test set.
	"""
	X_train, X_test, Y_train, Y_test = load_data(data_file)
	init_accuracy = check_accuracy( model, X_test, Y_test )

	# initialise the learning procedure
	model.fit( X_train, Y_train, epochs = 5 )
	
	# measure the time taken by a fixed number of iterations
	t0 = time.time()
	model.fit( X_train, Y_train, epochs = config.N_test )
	time_per_epoch = ( time.time() - t0 ) / ( config.N_test * config.secs_in_hr )
	
	N = int( time_in_hrs / time_per_epoch )
	hist = model.fit( X_train, Y_train, epochs = N, validation_data = ( X_test, Y_test ) )
	final_accuracy = check_accuracy( model, X_test, Y_test)
	aux.log('Initial accuracy: {}'.format( str( init_accuracy ) ) )
	aux.log('Accuracy after training: {}'.format( str( final_accuracy ) ) )
	return hist, init_accuracy, final_accuracy

def generate_params(learning_rate = 0.001, layer = 'GRU', units = 64, dropout = 0.2, recurrent_dropout = 0.2):
	"""
	Generates a default set of parameters.
	"""
	params = {}
	params['learning_rate'] = learning_rate
	params['layer'] = layer
	params['units'] = units
	params['dropout'] = dropout
	params['recurrent_dropout'] = recurrent_dropout
	return params

def plot_performance( model_name, history ):
	"""
	Given the history of a training process generates a plot of the loss and accuracy as a function of time, which is saved as an external file.
	"""
	fig, axs = plt.subplots(2)
	fig.suptitle('Training performance for {}'.format(model_name))
	axs[0].plot( history['loss'], label = 'Train set loss' )
	axs[0].plot( history['val_loss'], label = 'Test set loss' )
	axs[0].legend( loc = 'right' )
	# the accuracy plotted here is some kind of average, I am not sure if we should include it
	axs[1].plot( history['sparse_categorical_accuracy'], label = 'Train set accuracy' )
	axs[1].plot( history['val_sparse_categorical_accuracy'], label = 'Test set accuracy' )
	axs[1].legend( loc = 'right' )
	fig.savefig('results/{}.png'.format(model_name))
	plt.close()

def explore_model( model_name, input_shape, params, time_in_hrs = 1/60 ):
	"""
	Creates a model according to the specification, trains it for a specified amount of time
	(specified in hours) and evaluates the results on the test set. The results are plotted as well as saved in the database.
	"""
	if input_shape[1] in config.emb_dims:
		data_file = 'data/data{}d.npz'.format( input_shape[1] )
		model = create_model(input_shape, params)
		model.save('results/' + model_name + '_init.h5')
		hist, init_accuracy, final_accuracy = train_and_evaluate_model(model, data_file, time_in_hrs)
		model.save('results/' + model_name + '_final.h5')
		# Plot the training performance
		plot_performance( model_name, hist.history )
		# Store information about the training in the database
		coll, client = aux.get_collection('results')
		record = params
		record['model_name'] = model_name
		record['init_accuracy'] = init_accuracy
		record['final_accuracy'] = final_accuracy
		record['accuracy_increase'] = final_accuracy - init_accuracy

		if init_accuracy > 0:
			record['accuracy_ratio'] = final_accuracy / init_accuracy
		else:
			record['accuracy_ratio'] = config.max_accuracy

		record['training_time'] = time_in_hrs
		coll.insert_one( record )
		client.close()
		return hist
		
	else:
		aux.log('The embedding dimension must be chosen from the following list: {}'.format(emb_dims))
