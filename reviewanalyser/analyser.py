"""
This file contains functions related to analysing the reviews.
"""

import reviewanalyser.auxiliary_functions as aux
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, GRU

filename = '/home/jedrek/software/ML/glove.6B/glove.6B.50d.txt.short'
emb_dict = {}

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

def get_input_data(words = 10, emb_dim = 50, quality = 0.5):
	"""
	Returns an embedding of all the reviews from the database with at most a certain number of words and satisfying the quality threshold.
	"""
	client = aux.get_client()
	coll = client['ReviewAnalyser']['reviews']
	results = coll.find( { 'words': { '$lt': words }, 'quality': { '$gt': quality } } )
	count = coll.count_documents( { 'words': { '$lt': words } } )
	X = np.zeros( [ count, words, emb_dim ] )
	y = []
	
	j = 0

	for r in results:
		review_emb = convert_review( r['content'], words )
		
		if review_emb is not None:
			X[j, :, :] = review_emb
			# subtract 1 to ensure that the scores are in the range [0, 1, ..., 9]
			y.append( r['score'] - 1 )
			j += 1
	
	client.close()
	
	Y = np.array(y)
	
	return X[:j, : :], Y

def get_embedding_dict(filename):
	emb_dict = {}

	with open(filename, 'r') as csvfile:
		for line in csvfile:
			vals = line.split()
			word = vals[0]
			vect = np.array( vals[1:], dtype = 'float32' )
			emb_dict[word] = vect
	
	return emb_dict

def convert_review(review, length):
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

def create_model(input_shape):
	model = Sequential()
	model.add( GRU(64, dropout = 0.2, recurrent_dropout = 0.2, input_shape = input_shape ) )
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

def predict_rating(model, review, length):
	x = convert_review( review, length )
	y = model.predict( x.reshape( [1, x.shape[0], x.shape[1] ] ) )
	print( np.round(y, 3) )

"""
Things below must be properly cleaned up.
"""

"""
emb_dict = get_embedding_dict(filename)

X, Y = get_input_data(70, 50, 0.5)
model = create_model( ( X.shape[1], X.shape[2] ) )

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.15 )
model.fit( X_train, Y_train, epochs = 50 )
"""
#model.fit( X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100 )

review1 = 'This is a truly fantastic movie and I would like to watch it again. Stefan is a great director, noone can compete with him!'
review2 = 'This is an ok movie, but I have seen better. Moreover, it gets quite boring towards the end.'
review3 = 'This is a completely terrible movie, I have never seen anything worst in my life.'
