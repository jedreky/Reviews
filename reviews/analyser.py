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
def get_input_shape(params):
	if params['sentence_based']:
		input_shape = ( params['max_sentences'], params['max_words_per_sentence'], params['emb_dim'] )
	else:
		input_shape = ( params['max_words'], params['emb_dim'] )
	
	return input_shape

def generate_filter( criteria ):
	filt = []
	
	# add max_words criterion
	if 'max_words' in criteria:
		filt.append( { '$match': { 'words': { '$lte': criteria['max_words'] } } } )

	# add max_sentences criterion
	if 'max_sentences' in criteria:
		filt.append( { '$match': { 'sentences': { '$lte': criteria['max_sentences'] } } } )

	# add max_words_per_sentence criterion
	if 'max_words_per_sentence' in criteria:
		filt.append( { '$match': { 'words_per_sentence': { '$lte': criteria['max_words_per_sentence'] } } } )

	# add votes criterion
	if 'votes' in criteria:
		filt.append( { '$match': { 'quality': { '$gte': criteria['votes'] } } } )

	# add quality criterion
	if 'quality' in criteria:
		filt.append( { '$match': { 'quality': { '$gte': criteria['quality'] } } } )

	return filt

def check_score_distribution( client, criteria = {} ):
	"""
	From the processed reviews select the ones satisfying certain criteria and print the distribution of scores.
	"""
	coll = client[config.database_name]['reviews']

	pipeline = generate_filter(criteria)

	# group the reviews according to score and count them
	pipeline.append( { '$group': { '_id': '$score', 'count': { '$sum': 1 } } } )
	pipeline.append( { '$sort': { 'count': pymongo.DESCENDING } } )
	results = coll.aggregate( pipeline )

	for r in results:
		print(r)

def convert_text(text, length, padding, emb_dim, emb_dict):
	"""
	Converts a list of words into its embedding of a fixed length.
	"""
	
	text_emb = []
	text_emb_pad = None

	# construct the embedding by appending vectors corresponding to every word found in the dictionary
	for word in text.split():
		if word in emb_dict:
			text_emb.append( emb_dict[word] )
		else:
			aux.log('Warning: Word "{}" missing'.format(word))

	# check if the length does not exceed the limit
	if len(text_emb) <= length:
		# check if the length is non-zero
		if len(text_emb) > 0:
			# apply appropriate padding, so that text_emb_pad.shape = ( length, emb_dim )
			if padding == 'pre':
				text_emb_pad = np.pad( np.array(text_emb), ( ( length - len(text_emb) , 0), (0, 0) ) )
			elif padding == 'post':
				text_emb_pad = np.pad( np.array(text_emb), ( ( 0,  length - len(text_emb) ), (0, 0) ) )
			else:
				aux.log('Error: Invalid padding choice.')
		else:
			text_emb_pad = np.zeros( (length, emb_dim) )
		
		#TODO: assert shape of text_emb_pad ?

	else:
		aux.log('Error: Specified text is too long.')

	return text_emb_pad

def embed_reviews( X, params ):
	# get the embedding dictionary
	emb_dict = aux.get_emb_dict( params['emb_dim'] )

	# compute input shape and define empty array
	input_shape = get_input_shape(params)
	X_emb = np.zeros( (len(X), ) + input_shape )

	if params['sentence_based']:
		# iterate over reviews
		for j in range(len(X)):
			# split each review into sentences
			sentences = X[j].split('.')

			# iterate over non-empty sentences
			for k in range(len(sentences)):
				if sentences[k]:
					X_emb[j, k, :, :] = convert_text(sentences[k], params['max_words_per_sentence'], params['padding'], params['emb_dim'], emb_dict)
	else:
		# iterate over reviews
		for j in range(len(X)):
			# replace full stops by white spaces
			text = X[j].replace('.',' ')
			# compute the embedding
			X_emb[j, :, :] = convert_text(text, params['max_words'], params['padding'], params['emb_dim'], emb_dict)

	return X_emb

def generate_input_data(client, filename, N_reviews, criteria = {}):
	"""
	Constructs a balanced subset of reviews satisfying the selection criteria, generates input data
	(in several flavours) and stores it in .npz files.
	"""
	
	aux.log('Extracting reviews satisfying the following selection criteria:')

	coll = client[config.database_name]['reviews']

	filt = generate_filter( criteria )
	
	X = []
	Y = []

	# iterate over all the scores
	for score in range(1, 11):
		score_criterion = [ { '$match': { 'score': score } } ]
		results = coll.aggregate( filt + score_criterion + [{'$count': 'count'}] )

		# check if a sufficient number of reviews is found for each score
		# TODO: take care of empty result set
		if results.next()['count'] >= N_reviews:
			results = coll.aggregate( filt + score_criterion )

			for j in range(N_reviews):
				r = results.next()
				X.append( r['content'] )
				Y.append( r['score'] )

		else:
			aux.log('Too few reviews found for score: {}'.format(score))

	# shuffle both lists
	X, Y = sklearn.utils.shuffle(X, Y)
	# split the data into test and train sets
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y, test_size = 0.15 )
	Y_train = np.array(Y_train)
	Y_test = np.array(Y_test)
	
	params_template = {}
	
	for p in ('max_words', 'max_sentences', 'max_words_per_sentence'):
		params_template[p] = criteria[p]

	# iterate over all types of input data
	for emb_dim in config.emb_dims:
		params = params_template.copy()
		params['emb_dim'] = emb_dim

		for sentence_based in (True, False):
			params['sentence_based'] = sentence_based

			for padding in ('pre', 'post'):
				params['padding'] = padding
				X_train_emb = embed_reviews( X_train, params )
				X_test_emb = embed_reviews( X_test, params )
				f = 'input_data/{}-{}-{}d-{}.npz'.format( filename, sentence_based, emb_dim, padding )
				print(f)
				# save input data to an .npz file
				with open(f, 'wb') as data_file:
					np.savez(data_file, X_train = X_train_emb, X_test = X_test_emb, Y_train = Y_train, Y_test = Y_test)

	return X_train, Y_train

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

	params['input_shape'] = get_input_shape(params)

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
	# set the type of the main RNN units
	if params['RNN_type'] == 'GRU':
		RNN_layer = tf.keras.layers.GRU( params['RNN_units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout'] )
	elif params['RNN_type'] == 'LSTM':
		RNN_layer = tf.keras.layers.LSTM( params['RNN_units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout'] )
	else:
		aux.log('Warning: invalid value of the the RNN_type parameter.')

	# set the final layer
	if params['predictor'] == 'numerical':
		final_layer = tf.keras.layers.Dense(1, activation = None)
		loss = tf.keras.losses.MeanSquaredError()
	elif params['predictor'] == 'categorical':
		final_layer = tf.keras.layers.Dense(10, activation = 'softmax')
		loss = tf.keras.losses.SparseCategoricalCrossentropy()
	else:
		aux.log('Warning: invalid value of the predictor parameter.')

	if params['sentence_based']:
		# if sentence_based the sentences must be processed in parallel by the RNN
		inputs = tf.keras.Input( shape = params['input_shape'] )

		X_list = []

		# every sentence is fed to the RNN independently, the outputs are stored in a list
		for j in range( params['max_sentences'] ):
			X_list.append( RNN_layer( inputs[ :, j, :, : ] ) )

		# the list is converted to a TF object and reshaped
		X = tf.stack(X_list, axis = 1)
		req_shape = (-1, params['max_sentences'] * params['RNN_units'] )
		X = tf.reshape( X, shape = req_shape )

		# this is fed to the dense layers
		for units in params['Dense_units']:
			X = tf.keras.layers.Dense( units, activation = "relu" )(X)

		# and the final layer
		outputs = final_layer(X)
		model = tf.keras.Model( inputs = inputs, outputs = outputs )
		
	else:
		# if not sentence_based, the inputs are fed to RNN layers followed by the final layer
		model = tf.keras.models.Sequential()
		model.add( tf.keras.layers.InputLayer( input_shape = params['input_shape'] ) )
		model.add( RNN_layer )
		model.add( final_layer )

	# compile model
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
		aux.log('Error: the model is neither numerical nor categorical.')

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
	x = convert_text( review, length, emb_dict, 'post' )

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
