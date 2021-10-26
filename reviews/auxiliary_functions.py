"""
This file contains some auxiliary functions of the Reviews package.
"""

import datetime
import json
import numpy as np
import os
import pickle
import pymongo
import smtplib
import ssl

import reviews.config as config

#######################################################
# Short functions
#######################################################
def get_timestamp():
	"""
	Returns the current timestamp.
	"""
	timestamp = datetime.datetime.now()
	return timestamp.strftime('%H:%M:%S, %d.%m.%Y')

def log(string):
	"""
	Prints a timestamped message.
	"""
	print( '[{}] {}'.format( get_timestamp(), string ) )

def send_email(body):
	subject = 'Message from {}'.format(os.uname()[1])
	message = 'Subject: {}\n\n{}'.format( subject, body )

	with open('json_data/email_keys.json', 'r') as json_file:
		email_keys = json.load(json_file)
		context = ssl.create_default_context()
		
		with smtplib.SMTP_SSL( email_keys['smtp_server'], email_keys['port'], context = context ) as server:
			server.login( email_keys['sender_email'], email_keys['password'] )
			server.sendmail( email_keys['sender_email'], email_keys['receiver_email'], message)

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
	"""
	Converts a CSV text file into a pickle. Used to process the glove.6B files.
	"""
	emb_dict = {}

	with open( input_file + '.txt', 'r') as csv_file:
		for line in csv_file:
			vals = line.split()
			word = vals[0]
			vect = np.array( vals[1:], dtype = 'float32' )
			emb_dict[word] = vect
	
	output_file = input_file + '.pickle'

	with open(output_file, 'wb') as pickle_file:
		pickle.dump( emb_dict, pickle_file )

def pickle_emb_dict():
	"""
	Convert all the CSV glove files to pickle.
	"""
	for emb_dim in config.emb_dims:
		input_file = config.emb_dict_file.format(emb_dim)
		log('Pickling file: {}.txt'.format(input_file))
		convert_text_to_pickle(input_file)

def get_emb_dict( emb_dim ):
	"""
	Returns the embedding dictionary of specified dimension.
	"""
	filename = config.emb_dict_file.format(emb_dim) + '.pickle'

	with open(filename, 'rb') as pickle_file:
		emb_dict = pickle.load( pickle_file )
		return emb_dict

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
	client = None

	with open('json_data/mongo_keys.json', 'r') as json_file:
		mongo_keys = json.load(json_file)
		client = pymongo.MongoClient( username = mongo_keys['user'], password = mongo_keys['password'] )

	return client

def check_for_duplicates(client, collection, field, remove_duplicates = False):
	"""
	Checks whether in a given collection there are entries with identical values of the field.
	If specified, remove all but one.
	"""
	coll = client[config.database_name][collection]
	
	count_by_id = { '$group': { '_id': '$' + field, 'count': { '$sum': 1 } } }
	
	pipeline = [ count_by_id ]

	results = coll.aggregate( pipeline )
	
	duplicate_count = 0

	for r in results:
		if r['count'] > 1:
			movie_id = r['_id']
			aux.log('Duplicates found for: {} = {}'.format(field, movie_id))

			if remove_duplicates:
				# find all the duplicates
				records = coll.find( {'movie_id': movie_id} )
				# save the first record to add it back later
				record = records[0]
				# remove the _id key of the record (we do not need it)
				del( record['_id'] )
				# remove all the duplicates
				coll.delete_many( {'movie_id': movie_id} )
				# add the record back in
				coll.insert_one( record )
				
				aux.log('Duplicates removed for: {} = {}'.format(field, r['_id']))

			duplicate_count += 1
	
	if duplicate_count == 0:
		aux.log('No duplicates founds.')
