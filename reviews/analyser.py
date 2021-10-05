"""
This file contains functions related to analysing the reviews.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pymongo
import sklearn.model_selection
import sklearn.utils
import time

import tensorflow
from tensorflow.keras.layers import Dense, GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import reviews.auxiliary_functions as aux
import reviews.config as config

############################################################
# Functions related to data verification and preparation
############################################################
def check_score_distribution( max_words = None, quality = 0 ):
	"""
	From the reviews database select the reviews satisfying certain criteria and print the distribution of scores.
	"""
	coll, client = aux.get_collection('reviews')

	pipeline = []

	# add length criterion
	if max_words is not None:
		pipeline.append( { '$match': { 'words': { '$lte': max_words } } } )

	# add quality criterion
	if quality > 0:
		pipeline.append( { '$match': { 'quality': { '$gte': quality } } } )

	# group the reviews according to score and count them
	pipeline.append( { '$group': { '_id': '$score', 'count': { '$sum': 1 } } } )
	pipeline.append( { '$sort': { 'count': pymongo.DESCENDING } } )
	results = coll.aggregate( pipeline )

	for r in results:
		print(r)

	client.close()

def convert_review(review, length, emb_dict):
	"""
	Converts a review into its embedding.
	"""
	# convert to lower case
	review = review.lower()
	
	# define a tuple of symbols to be removed
	symbols_to_remove = ( ':' , ',' , '.', '!', '-', '(', ')' )
	
	# remove symbols
	for symbol in symbols_to_remove:
		review = review.replace( symbol, ' ' )

	# split the string into words
	words = review.split()
	
	review_emb = []

	# construct the embedding by appending vectors corresponding to every word found in the dictionary
	for word in words:
		if word in emb_dict:
			review_emb.append( emb_dict[word] )
#		else:
#			#print('Word "{}" missing'.format(word))

	# check if the review matches the length criteria
	if len(review_emb) <= length:
		# if yes, apply appropriate padding
		review_emb_pad = np.pad( np.array(review_emb), ( ( length - len(review_emb) , 0), (0, 0) ) )
	else:
		# if no, set the return variable to None
		review_emb_pad = None

	return review_emb_pad

def get_input_data(n, max_words, emb_dim, quality):
	"""
	Returns an embedding of reviews from the database that satisfy certain 	criteria (number of words and quality).
	To ensure that we are training on a balanced dataset we choose the same number of reviews (denoted by n) with each score.
	In the last step the reviews are randomly permuted.
	"""
	aux.log('Extracting input data for: n = {}, max_words = {}, emb_dim = {}, quality = {}'.format(n, max_words, emb_dim, quality))

	coll, client = aux.get_collection('reviews')

	# initialise empty arrays
	X = np.zeros( [ 10 * n, max_words, emb_dim ] )
	Y = np.zeros( [ 10 * n, 1] )

	# get embedding dictionary
	emb_dict = aux.get_emb_dict( emb_dim )

	# initialise the total counter of reviews
	count_total = 0

	# iterate over all the scores
	for score in range(1, 11):
		# count the number of reviews satisfying the criteria and extract them
		count = coll.count_documents( { 'words': { '$lte': max_words }, 'quality': { '$gte': quality }, 'score': score } )
		results = coll.find( { 'words': { '$lte': max_words }, 'quality': { '$gte': quality }, 'score': score } )
		
		# initialise the counter for a given score
		count_per_score = 0
		
		# initialise the current position
		k = 0

		# loop over reviews until either we obtain the desired number or run out of reviews
		while count_per_score < n and k < count:
			review_emb = convert_review( results[k]['content'], max_words, emb_dict )
			k += 1

			# if a valid embedding is returned add it to the dataset
			if review_emb is not None:
				X[count_total + count_per_score, :, :] = review_emb
				# subtract 1 to ensure that the scores are in the range [0, 1, ..., 9]
				Y[count_total + count_per_score] = results[k]['score'] - 1
				count_per_score += 1
		
		# update the total counter
		count_total += count_per_score

	client.close()
	
	# truncate the empty rows
	X = X[:count_total, :, :]
	Y = Y[:count_total]
	# shuffle the dataset
	X, Y = sklearn.utils.shuffle(X, Y)

	# check if a sufficient number of reviews has been found
	if count_total == 10 * n:
		aux.log('A sufficient number of reviews found.')
	else:
		aux.log('Insufficient reviews, the resulting matrices are smaller than asked for.')
	
	return X, Y

def generate_input_data_fixed_emb_dim(filename, n, max_words, emb_dim, quality):
	"""
	Generates input data for a fixed embedding dimension and saves it to an .npz file.
	"""
	# get input data from database
	X, Y = get_input_data(n, max_words, emb_dim, quality)
	# split input data into train and test sets
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y, test_size = 0.15 )

	# save input data to an .npz file
	with open('input_data/{}d.npz'.format( filename ), 'wb') as data_file:
		np.savez(data_file, X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test)

def generate_input_data(filename, n = 15, max_words = 150, quality = 0.5):
	"""
	Generates input data for all valid embedding dimensions and saves them to .npz files.
	"""
	# iterate over all valid embedding dimensions
	for emb_dim in config.emb_dims:
		generate_input_data_fixed_emb_dim( filename + str(emb_dim), n, max_words, emb_dim, quality )

def load_data(data_file):
	"""
	Loads data from an .npz file.
	"""
	data = np.load(data_file)
	return data['X_train'], data['X_test'], data['Y_train'], data['Y_test']
############################################################
# Functions related to ML models
############################################################
def generate_params(learning_rate = 0.001, layer = 'GRU', units = 64, dropout = 0.2, recurrent_dropout = 0.2, predictor = 'numerical'):
	"""
	Generates a default set of parameters.
	"""
	params = {}
	params['learning_rate'] = learning_rate
	params['layer'] = layer
	params['units'] = units
	params['dropout'] = dropout
	params['recurrent_dropout'] = recurrent_dropout
	params['predictor'] = predictor

	return params

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
		aux.log('Warning: invalid value of the the layer parameter.')

	# it seems that numerical makes much more sense
	if params['predictor'] == 'numerical':
		model.add( Dense(1, activation = None) )
		model.compile( loss = tensorflow.keras.losses.MeanSquaredError(), optimizer = Adam( learning_rate = params['learning_rate'] ) )
	elif params['predictor'] == 'categorical':
		model.add( Dense(10, activation = 'softmax') )
		model.compile( loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(), optimizer = Adam( learning_rate = params['learning_rate'] ) )
	else:
		aux.log('Warning: invalid value of the the predictor parameter.')

	return model

def check_accuracy(model, X, Y, err = 1):
	"""
	Returns the fraction of examples for which the returned score is close to the true score up to the specified additive error.
	"""
	# look at the number of units in the final layer determine whether the model is numerical or categorical
	# based on this define how to interpret the prediction
	if model.layers[1].units == 1:
		interpret = lambda x : x
	elif model.layers[1].units == 10:
		interpret = lambda x : np.argmax(x)
	else:
		aux.log('Warning: the model is neither numerical nor categorical.')

	# initialise the counter for good enough predictions
	count = 0

	for j in range( X.shape[0] ):
		Y_pred = interpret( model.predict( X[j].reshape( [1, X.shape[1], X.shape[2]] ) ) )

		if np.abs( Y_pred - Y[j] ) <= err:
			count += 1

	# compute the fraction of good enough predictions
	frac = float(count / X.shape[0])

	return frac

def predict_rating(model, review, length, emb_dim):
	"""
	Given a model and a review returns the prediction.
	"""
	emb_dict = aux.get_emb_dict( emb_dim )

	# convert the review into a vector
	x = convert_review( review, length, emb_dict )

	# if successful compute the prediction
	if x is not None:
		y = model.predict( x.reshape( [1, x.shape[0], x.shape[1] ] ) )
		print( np.round(y, 3) )

def train_model(model_name, model, data_file, time_in_secs):
	"""
	Trains the given model for a required amount of time and returns the training history.
	"""
	X_train, X_test, Y_train, Y_test = load_data(data_file)
	train_loss = []
	test_loss = []
	train_accuracy = []
	test_accuracy = []
	
	# compute the initial accuracy of the model
	train_accuracy.append( check_accuracy( model, X_train, Y_train ) )
	test_accuracy.append( check_accuracy( model, X_test, Y_test ) )

	# start the clock
	t0 = time.time()

	# keep training until we run out of time
	while time.time() - t0 < time_in_secs:
		# train the model for a fixed number of epochs
		# the history object stores the information about the loss on train and test sets
		hist = model.fit( X_train, Y_train, epochs = config.N_epochs, validation_data = ( X_test, Y_test ) )
		# save the model after each iteration
		model.save('results/{}/{}_final.h5'.format( model_name[0], model_name[1] ))
		# append the loss information to the lists
		train_loss += hist.history['loss']
		test_loss += hist.history['val_loss']
		# compute the accuracy and append it to the lists
		train_accuracy.append( check_accuracy( model, X_train, Y_train ) )
		test_accuracy.append( check_accuracy( model, X_test, Y_test ) )

	# store the training results in a single dictionary
	results = {}
	results['train_loss'] = train_loss
	results['test_loss'] = test_loss
	results['train_accuracy'] = train_accuracy
	results['test_accuracy'] = test_accuracy

	return results

def plot_performance( model_name, time_in_hrs, results ):
	"""
	Given the training information generates a plot of the loss and accuracy as a function of time and saves it as a .png file.
	"""
	# create a figure
	fig, axs = plt.subplots(2)
	
	# generate the time information
	if time_in_hrs >= 1:
		time_info = ' (~{} hrs)'.format(int(time_in_hrs))
	else:
		time_info = ''

	# set the plot title
	plot_title = 'Training performance for {}.{}'.format( model_name[0], model_name[1], int(time_in_hrs) ) + time_info
	fig.suptitle( plot_title )

	# plot the loss data
	axs[0].plot( results['train_loss'], label = 'Train set loss' )
	axs[0].plot( results['test_loss'], label = 'Test set loss' )
	axs[0].legend( loc = 'right' )

	# plot the accuracy data
	axs[1].plot( results['train_accuracy'], label = 'Train set accuracy' )
	axs[1].plot( results['test_accuracy'], label = 'Test set accuracy' )
	axs[1].legend( loc = 'right' )

	# save the figure
	fig.savefig('results/{}/{}.png'.format( model_name[0], model_name[1] ))
	plt.close()

def explore_model( model_family, input_shape, params, time_in_hrs = 1/60 ):
	"""
	Creates a model according to the specification, trains it for a specified amount of time (specified in hours) and evaluates the results
	on the test set. The results are then plotted, saved in an .npz file and in the database.
	"""
	# check if a valid embedding dimension is specified
	if input_shape[1] in config.emb_dims:

		# check if the subdirectory for storing results exists
		if not os.path.exists('results/{}'.format(model_family)):
			# if not, create one
			aux.log('Directory {} not present'.format(model_family))
			os.mkdir('results/{}'.format(model_family))

		# determine the full name of the current model (based on the number of models from this family already present in the database)
		coll, client = aux.get_collection('results')
		count = coll.count_documents( { 'model_family': model_family } ) + 1
		model_name = ( model_family, count )

		# create a model and save it
		model = create_model(input_shape, params)
		model.save('results/{}/{}_init.h5'.format( model_name[0], model_name[1] ))
		
		# train the model for a specified amount of time
		data_file = 'input_data/data{}d.npz'.format( input_shape[1] )
		results = train_model(model_name, model, data_file, time_in_hrs * config.secs_in_hr )

		# plot the training performance
		plot_performance( model_name, time_in_hrs, results )
		
		# save the results to an .npz file
		with open('results/{}/{}.npz'.format( model_name[0], model_name[1] ), 'wb') as data_file:
			np.savez(data_file, train_loss = results['train_loss'], test_loss = results['test_loss'], train_accuracy = results['train_accuracy'], test_accuracy = results['test_accuracy'])
		
		# store the training information in the database
		record = params
		record['model_family'] = model_name[0]
		record['model_id'] = model_name[1]
		record['init_accuracy'] = results['test_accuracy'][0]
		record['final_accuracy'] = results['test_accuracy'][-1]
		record['training_time'] = time_in_hrs
		coll.insert_one( record )
		client.close()

		# this is currently optional, might be removed in the future
		return results

	else:
		aux.log('The embedding dimension must be chosen from the following list: {}'.format(emb_dims))
