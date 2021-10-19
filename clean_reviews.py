"""
This will later be turned into a pre-processing function for reviews (so that we can compare
sentence-based models against the review-based ones.
"""

import numpy as np
import re
import tensorflow as tf

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

def sanitise_review(review):
	"""
	Sanitises a review.
	"""
	review = review.lower()

	# expand shortenings
	review = review.replace("n't", ' not')
	review = review.replace("'ll", ' will')
	review = review.replace("'ve", ' have')
	review = review.replace("'re", ' are')
	review = review.replace("'m", ' am')
	
	s_list = ( 'that', 'what', 'he', 'she', 'it')
	for word in s_list:
		review = review.replace( word + "'s" , word + ' is')

	# change all end-of-sentence characters to a full stop
	EOS_list = ( '.', '!', '?' )
	for char in EOS_list:
		review = review.replace( char, '.')

	# reduce sequences of dots
	review = re.sub('\.+', '.', review)
	
	# remove all characters except for letters, white spaces and full stops
	review = re.sub('[^a-z .]', '', review)
	
	return review
	

max_words = 150
emb_dim = 50

coll, client = aux.get_collection('reviews')

results = coll.find( { 'words': { '$lte': max_words } } )

for j in range(1020):
	print( sanitise_review( results[j]['content'] ) + '\n' )
