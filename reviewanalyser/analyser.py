"""
This file contains functions related to analysing the reviews.
"""

import reviewanalyser.auxiliary_functions as aux
import csv
import numpy as np
import sklearn.model_selection
import sklearn.utils
import pickle
import tensorflow as tf
import time

from keras.models import Sequential
from keras.layers import Dense, GRU

def check_score_distribution():
	"""
	Checks the distribution of scores in the reviews.
	"""
	client = aux.get_client()
	coll = client['ReviewAnalyser']['reviews']
	
	count_by_score = { '$group': { '_id': '$score', 'count': { '$sum': 1 } } }
	pipeline = [ count_by_score ]
	results = coll.aggregate( pipeline )

	for r in results:
		print(r)

	client.close()

def get_input_data(n = 10, max_words = 10, emb_dim = 50, quality = 0.5):
	"""
	Returns an embedding of reviews from the database that satisfy certain criteria	(maximum number of words and minimum quality).
	To ensure that we are training on an approximately balanced set we choose the same number of reviews (denoted by n) with each grade.
	"""
	client = aux.get_client()
	coll = client['ReviewAnalyser']['reviews']

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

	if j == 10 * n:
		aux.log('A sufficient number of reviews found.')
	else:
		aux.log('Insufficient reviews, the resulting matrices are smaller than asked for.')
	
	return X, Y

def convert_review(review, length, emb_dict):
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

def create_model(input_shape, units = 64):
	model = Sequential()
	model.add( GRU(units, dropout = 0.2, recurrent_dropout = 0.2, input_shape = input_shape ) )
	model.add( Dense(10, activation = 'softmax') )
	model.compile( loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = 'adam', metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] )
	return model

def check_accuracy(model, X, Y, err = 1):
	"""
	Returns the fraction of examples for which the returned score is close to the true score.
	"""
	count = 0
	for j in range( X.shape[0] ):
		Y_pred = np.argmax( model.predict( X[j].reshape( [1, X.shape[1], X.shape[2]] ) ) )
		if np.abs( Y_pred - Y[j] ) <= err:
			count += 1
	
	return float(count / X.shape[0])

def predict_rating(model, review, length, emb_dim):
	with open('glove.6B/glove.6B.{}d.pickle'.format( str(emb_dim) ), 'rb') as pickle_file:
		emb_dict = pickle.load( pickle_file )

	x = convert_review( review, length, emb_dict )
	y = model.predict( x.reshape( [1, x.shape[0], x.shape[1] ] ) )
	print( np.round(y, 3) )

def train_and_evaluate_model(model, X_train, X_test, Y_train, Y_test, time_in_mins = 60):
	"""
	Trains the given model for a required amount of time and evaluates its performance.
	"""
	aux.log('Initial accuracy: {}'.format( str( check_accuracy( model, X_test, Y_test) ) ) )

	# initialise the learning procedure
	model.fit( X_train, Y_train, epochs = 5 )
	
	# measure the time taken by a fixed number of iterations
	N_test = 5
	t0 = time.time()
	model.fit( X_train, Y_train, epochs = N_test )
	time_per_epoch = ( time.time() - t0 ) / ( N_test * 60 )
	
	N = int( time_in_mins / time_per_epoch )
	model.fit( X_train, Y_train, epochs = N )
	aux.log('Accuracy after training: {}'.format( str( check_accuracy( model, X_test, Y_test) ) ) )

def start( n = 30, max_words = 130 ):
	X, Y = get_input_data(n, max_words, 50, 0.5)
	model = create_model( ( X.shape[1], X.shape[2] ) )
	
	return X, Y, model


"""
Things below must be properly cleaned up.
"""


"""
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y, test_size = 0.15 )
model.fit( X_train, Y_train, epochs = 50 )

#model.fit( X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100 )

review1 = 'This is a truly fantastic movie and I would like to watch it again. Stefan is a great director, noone can compete with him!'
review2 = 'This is an ok movie, but I have seen better. Moreover, it gets quite boring towards the end.'
review3 = 'This is a completely terrible movie, I have never seen anything worst in my life.'
"""
