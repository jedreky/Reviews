"""
This file contains some auxiliary functions of the ReviewAnalyser.
"""

import datetime
import json
import numpy as np
import pickle
import pymongo

#######################################################
# Short functions
#######################################################
def get_timestamp():
	"""
	Returns the current timestamp.
	"""
	timestamp = datetime.datetime.now()
	return timestamp.strftime('%H:%M:%S, %d.%m.%Y')

def log(str):
	"""
	Prints a timestamped message.
	"""
	print( get_timestamp() + ': ' + str )

def convert_to_int(string):
	return int(string.replace(',', ''))

def sanitise_text(string):
	symbols_to_remove = ( '\n', '<br/>' )
	string = string.replace('&#39;', "'")
	string = string.replace('&quot;', '"')
	string = string.replace('&amp;', '&')
	
	for symbol in symbols_to_remove:
		string = string.replace(symbol, '')

	return string

def convert_text_to_pickle( input_file ):
	emb_dict = {}

	with open(input_file, 'r') as csv_file:
		for line in csv_file:
			vals = line.split()
			word = vals[0]
			vect = np.array( vals[1:], dtype = 'float32' )
			emb_dict[word] = vect
	
	output_file = input_file[:-3] + 'pickle'
	with open(output_file, 'wb') as pickle_file:
		pickle.dump( emb_dict, pickle_file )

def get_random_sleep_time(min_val = 3, ave_val = 8, alpha = 4):
	"""
	Returns a random sleep time according to the parameters.
	"""
	rng = np.random.default_rng()
	return np.max( [ min_val, ave_val + alpha * rng.standard_normal() ] )


def get_client():
	"""
	Returns a MongoClient.
	"""
	with open('mongo_keys.json', 'r') as json_file:
		mongo_keys = json.load(json_file)
		client = pymongo.MongoClient( username = mongo_keys[0], password = mongo_keys[1] )
		return client