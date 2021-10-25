"""
This file contains functions related to processing and analysing the reviews.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pymongo
import re
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf
import time

import reviews.auxiliary_functions as aux
import reviews.config as config

############################################################
# Functions related to data verification and preparation
############################################################
def check_score_distribution( client, max_words = None, max_sentences = None, max_words_per_sentence = None, quality = 0 ):
	"""
	From the processed reviews select the ones satisfying certain criteria and print the distribution of scores.
	"""
	coll = coll = client[config.database_name]['reviews']

	pipeline = []

	# add max_words criterion
	if max_words is not None:
		pipeline.append( { '$match': { 'words': { '$lte': max_words } } } )

	# add max_sentences criterion
	if max_sentences is not None:
		pipeline.append( { '$match': { 'sentences': { '$lte': max_sentences } } } )

	# add max_words_per_sentence criterion
	if max_words_per_sentence is not None:
		pipeline.append( { '$match': { 'words_per_sentence': { '$lte': max_words_per_sentence } } } )

	# add quality criterion
	if quality > 0:
		pipeline.append( { '$match': { 'quality': { '$gte': quality } } } )

	# group the reviews according to score and count them
	pipeline.append( { '$group': { '_id': '$score', 'count': { '$sum': 1 } } } )
	pipeline.append( { '$sort': { 'count': pymongo.DESCENDING } } )
	results = coll.aggregate( pipeline )

	for r in results:
		print(r)

def sanitise_review_content(content):
	"""
	Sanitises the content of a review by removing non-character symbols,
	removing some abbreviations and unifying all end-of-sentence characters.
	"""
	content = content.lower()

	# expand shortenings
	content = content.replace("n't", ' not')
	content = content.replace("'ll", ' will')
	content = content.replace("'ve", ' have')
	content = content.replace("'re", ' are')
	content = content.replace("'m", ' am')

	s_list = ( 'that', 'what', 'he', 'she', 'it')
	for word in s_list:
		content = content.replace( word + "'s" , word + ' is')

	# change all end-of-sentence characters to a full stop
	EOS_list = ( '.', '!', '?' )
	for char in EOS_list:
		content = content.replace( char, '.')

	# reduce sequences of dots
	content = re.sub('\.+', '.', content)

	# remove all characters except for letters, white spaces and full stops
	content = re.sub('[^a-z .]', '', content)

	return content

def process_raw_reviews(client):
	"""
	Process all the reviews from the raw_reviews database and store them in the reviews database.
	"""
	coll_raw = client[config.database_name]['raw_reviews']

	results = coll_raw.find()

	coll_reviews = client[config.database_name]['reviews']

	for record in results:
		# remove unnecessary fields
		record.pop('_id')
		record.pop('chars')

		# sanitise the content
		content = sanitise_review_content( record['content'] )
		record['content'] = content

		# split the content into sentences
		sentences = content.split('.')

		# compute the length (in words) of each sentence
		lengths = list( map( lambda x : len( x.split() ), sentences ) )

		# count the number of sentences, total number of words and the maximal number of words per sentence
		record['sentences'] = sum( x > 0 for x in lengths )
		record['words'] = sum(lengths)
		record['words_per_sentence'] = max( lengths )
		coll_reviews.insert_one( record )

def convert_review(review, length, emb_dict, padding):
	"""
	Converts a review into its embedding.
	"""
	# convert to lower case
	review = review.lower()
	
	# define a tuple of symbols to be removed
	symbols_to_remove = ( ':' , ',' , '.', '!', '-', '(', ')' )
	
	# replace symbols by white space
	for symbol in symbols_to_remove:
		review = review.replace( symbol, ' ' )
	
	print(review)
	
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
		# note that review_emb.shape = ( max_words, emb_dim )
		if padding == 'pre':
			review_emb_pad = np.pad( np.array(review_emb), ( ( length - len(review_emb) , 0), (0, 0) ) )
		elif padding == 'post':
			review_emb_pad = np.pad( np.array(review_emb), ( ( 0,  length - len(review_emb) ), (0, 0) ) )

	else:
		# if no, set the return variable to None
		review_emb_pad = None

	return review_emb_pad

def get_input_data(client, n, max_words, emb_dim, quality, padding):
	"""
	Returns an embedding of reviews from the database that satisfy certain criteria (number of words and quality).
	To ensure that we are training on a balanced dataset we choose the same number of reviews (denoted by n) with each score.
	In the last step the reviews are randomly permuted.
	"""
	aux.log('Extracting input data for: n = {}, max_words = {}, emb_dim = {}, quality = {}, padding = {}'.format(n, max_words, emb_dim, quality, padding))

	coll = client[config.database_name]['reviews']

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
			review_emb = convert_review( results[k]['content'], max_words, emb_dict, padding )
			k += 1

			# if a valid embedding is returned add it to the dataset
			if review_emb is not None:
				X[count_total + count_per_score, :, :] = review_emb
				# subtract 1 to ensure that the scores are in the range [0, 1, ..., 9]
				Y[count_total + count_per_score] = results[k]['score'] - 1
				count_per_score += 1
		
		# update the total counter
		count_total += count_per_score
	
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

def generate_input_data_fixed_emb_dim(filename, n, max_words, emb_dim, quality, padding):
	"""
	Generates input data for a fixed embedding dimension and saves it to an .npz file.
	"""
	# get input data from database
	X, Y = get_input_data(n, max_words, emb_dim, quality, padding)
	# split input data into train and test sets
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y, test_size = 0.15 )

	# save input data to an .npz file
	with open('input_data/{}{}d-{}.npz'.format( filename, emb_dim, padding ), 'wb') as data_file:
		np.savez(data_file, X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test)

def generate_input_data(filename, n = 15, max_words = 150, quality = 0.5, padding = 'post'):
	"""
	Generates input data for all valid embedding dimensions and saves them to .npz files.
	"""
	# iterate over all valid embedding dimensions
	for emb_dim in config.emb_dims:
		generate_input_data_fixed_emb_dim( filename, n, max_words, emb_dim, quality, padding )

def load_data(data_file):
	"""
	Loads data from an .npz file.
	"""
	data = np.load(data_file)
	return data['X_train'], data['X_test'], data['Y_train'], data['Y_test']
############################################################
# Functions related to ML models
############################################################
def generate_params( sentence_based = False, RNN_type = 'GRU', RNN_units = 32, Dense_units = [16, 16, 16], predictor = 'numerical', learning_rate = 0.001, dropout = 0.2, recurrent_dropout = 0.2, max_words = 150, max_sentences = 30, max_words_per_sentence = 30, emb_dim = 50 ):
	"""
	Generates a complete set of parameters.
	"""
	params = {}
	params['learning_rate'] = learning_rate
	params['RNN_type'] = RNN_type
	params['RNN_units'] = RNN_units
	params['dropout'] = dropout
	params['recurrent_dropout'] = recurrent_dropout
	params['predictor'] = predictor
	params['sentence_based'] = sentence_based
	params['max_words'] = max_words
	params['max_sentences'] = max_sentences
	params['max_words_per_sentence'] = max_words_per_sentence
	params['emb_dim'] = emb_dim
	params['Dense_units'] = Dense_units
	
	if sentence_based:
		input_shape = ( params['max_sentences'], params['max_words_per_sentence'], params['emb_dim'] )
	else:
		input_shape = ( params['max_words'], params['emb_dim'] )
	
	params['input_shape'] = input_shape

	return params

def get_model_params(client, model_id):
	"""
	Extracts parameters of a given model from the database.
	"""
	# obtain the whole dictionary from the database
	coll = client[config.database_name]['results']
	params = coll.find_one( { 'batch_name': model_id[0], 'batch_counter': model_id[1] } )
	
	# remove the unwanted keys
	keys_to_remove = ( '_id', 'batch_name', 'batch_counter', 'init_accuracy', 'final_accuracy', 'training_time', 'parent_id' )
	for key in keys_to_remove:
		# first check if the key is present to avoid errors
		if key in params:
			params.pop(key)

	return params

def create_model(params):
	"""
	Creates a recurrent neural network according to the specified parameters.
	"""
	if params['RNN_type'] == 'GRU':
		RNN_layer = tf.keras.layers.GRU( params['RNN_units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout'] )
	elif params['RNN_type'] == 'LSTM':
		RNN_layer = tf.keras.layers.LSTM( params['RNN_units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout'] )
	else:
		aux.log('Warning: invalid value of the the RNN_type parameter.')

	if params['predictor'] == 'numerical':
		final_layer = tf.keras.layers.Dense(1, activation = None)
		loss = tf.keras.losses.MeanSquaredError()
	elif params['predictor'] == 'categorical':
		final_layer = tf.keras.layers.Dense(10, activation = 'softmax')
		loss = tf.keras.losses.SparseCategoricalCrossentropy()
	else:
		aux.log('Warning: invalid value of the predictor parameter.')

	if params['sentence_based']:
		inputs = tf.keras.Input( shape = params['input_shape'] )
		
		X_list = []
		
		for j in range( params['max_sentences'] ):
			X_list.append( RNN_layer( inputs[ :, j, :, : ] ) )

		X = tf.stack(X_list, axis = 1)
		req_shape = (-1, params['max_sentences'] * params['RNN_units'] )
		
		X = tf.reshape( X, shape = req_shape )
		
		for units in params['Dense_units']:
			X = tf.keras.layers.Dense( units, activation = "relu" )(X)
		
		outputs = final_layer(X)
		model = tf.keras.Model( inputs = inputs, outputs = outputs )
			
	else:
		model = tf.keras.models.Sequential()
		model.add( tf.keras.layers.InputLayer( input_shape = params['input_shape'] ) )
		model.add( RNN_layer )
		model.add( final_layer )

	model.compile( loss = loss, optimizer = tf.keras.optimizers.Adam( learning_rate = params['learning_rate'] ) )

	return model

def compute_accuracy(model, X, Y, err = 1):
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

def compute_mse(model, X, Y):
	"""
	For numerical models, compute the mean squared error manually.
	This is just as a double check of the existing MSE implementation.
	"""
	val = 0

	for j in range( X.shape[0] ):
		Y_pred = model.predict( X[j].reshape( [1, X.shape[1], X.shape[2]] ) )
		val += np.power( (Y_pred - Y[j]), 2 )

	mse = val / X.shape[0]

	return mse

def predict_rating(model, review, length, emb_dim):
	"""
	Given a model and a review returns the prediction.
	"""
	emb_dict = aux.get_emb_dict( emb_dim )

	# convert the review into a vector
	x = convert_review( review, length, emb_dict, 'post' )

	# if successful compute the prediction
	if x is not None:
		y = model.predict( x.reshape( [1, x.shape[0], x.shape[1] ] ) )
		print( np.round(y, 3) )

def train_model(model_id, model, data_file, time_in_secs):
	"""
	Trains the given model for a required amount of time, while saving at regular intervals, and returns the training history.
	"""
	X_train, X_test, Y_train, Y_test = load_data(data_file)
	train_loss = []
	test_loss = []
	train_accuracy = []
	test_accuracy = []
	
	# compute the initial accuracy of the model
	train_accuracy.append( compute_accuracy( model, X_train, Y_train ) )
	test_accuracy.append( compute_accuracy( model, X_test, Y_test ) )

	# start the clock
	t0 = time.time()

	# keep training until we run out of time
	while time.time() - t0 < time_in_secs:
		# train the model for a fixed number of epochs
		# the history object stores the information about the loss on train and test sets
		hist = model.fit( X_train, Y_train, epochs = config.N_epochs, validation_data = ( X_test, Y_test ) )
		# save the model after each iteration
		model.save('results/{}/{}_final.h5'.format( model_id[0], model_id[1] ))
		# append the loss information to the lists
		train_loss += hist.history['loss']
		test_loss += hist.history['val_loss']
		# compute the accuracy and append it to the lists
		train_accuracy.append( compute_accuracy( model, X_train, Y_train ) )
		test_accuracy.append( compute_accuracy( model, X_test, Y_test ) )

	# store the training results in a single dictionary
	results = {}
	results['train_loss'] = train_loss
	results['test_loss'] = test_loss
	results['train_accuracy'] = train_accuracy
	results['test_accuracy'] = test_accuracy

	return results

def plot_performance( model_id, time_in_hrs, results ):
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
	plot_title = 'Training performance for {}.{}'.format( model_id[0], model_id[1], int(time_in_hrs) ) + time_info
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
	fig.savefig('results/{}/{}.png'.format( model_id[0], model_id[1] ))
	plt.close()

#def setup_and_train_model( batch_name, input_shape, params, time_in_hrs = 1/60, initial_model = None ):
def setup_and_train_model( client, batch_name, params, time_in_hrs = 1/60):
	"""
	Sets up a model according to the specification, trains it for a specified amount of time (specified in hours) and
	evaluates the results on the test set. The results are then plotted, saved in an .npz file and in the database.
	"""
#	# check if a valid embedding dimension is specified
#	if input_shape[1] in config.emb_dims:

	# check if the subdirectory for storing results exists
	if not os.path.exists('results/{}'.format(batch_name)):
		# if not, create one
		aux.log('Directory {} not present, will create it now'.format(batch_name))
		os.mkdir('results/{}'.format(batch_name))

	# determine the full name of the current model (based on the number of models from this family already present in the database)
	coll = client[config.database_name]['results']
	batch_counter = coll.count_documents( { 'batch_name': batch_name } ) + 1
	model_id = ( batch_name, batch_counter )

	# create a model and save it
	if 'parent_id' in params:
		parent_id = params['parent_id']
		aux.log('Loading initial model and its parameters: {}.{}'.format( parent_id[0], parent_id[1] ) )
		model = tf.keras.models.load_model( 'results/{}/{}_final.h5'.format( parent_id[0], parent_id[1] ) )
		print('should not happen')
		params.update( get_model_params( parent_id ) )
	else:
		aux.log('No initial model specified, creating from scratch')
		model = create_model(params)

	model.save('results/{}/{}_init.h5'.format( model_id[0], model_id[1] ))
	
	# train the model for a specified amount of time
#	data_file = 'input_data/data{}d.npz'.format( params['input_shape'][1] )
	aux.log('Train model {}.{} for {} hours'.format( model_id[0], model_id[1], time_in_hrs ) )
	aux.log('Using data file: {}'.format( params['data_file'] ))
	results = train_model(model_id, model, params['data_file'], time_in_hrs * config.secs_in_hr )

	# plot the training performance
	plot_performance( model_id, time_in_hrs, results )
	
	# save the results to an .npz file
	with open('results/{}/{}.npz'.format( model_id[0], model_id[1] ), 'wb') as data_file:
		np.savez(data_file, train_loss = results['train_loss'], test_loss = results['test_loss'], train_accuracy = results['train_accuracy'], test_accuracy = results['test_accuracy'])

	# store the training information in the database
	params['batch_name'] = batch_name
	params['batch_counter'] = batch_counter
	params['train_loss'] = ( results['train_loss'][0], results['train_loss'][-1] )
	params['test_loss'] = ( results['test_loss'][0], results['test_loss'][-1] )
	params['train_accuracy'] = ( results['train_accuracy'][0], results['train_accuracy'][-1] )
	params['test_accuracy'] = ( results['test_accuracy'][0], results['test_accuracy'][-1] )
	params['training_time'] = time_in_hrs
	coll.insert_one( params )

	# this is currently optional, might be removed in the future
	return results

#	else:
#		aux.log('The embedding dimension must be chosen from the following list: {}'.format(emb_dims))
