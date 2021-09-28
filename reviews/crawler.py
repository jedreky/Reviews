"""
This file contains functions related to crawling the IMDB website, extracting reviews and storing them in a Mongo database.
"""

import reviewanalyser.auxiliary_functions as aux
import json
import numpy as np
import re
import requests
import time

def get_movies_from_genre(genre, n):
	"""
	Finds at least n new movies from the given genre and adds them to the database.
	"""
	movies = []

	client = aux.get_client()
	coll = client['ReviewAnalyser']['movies']

	k = 1
	while n > len(movies):
		url = 'https://www.imdb.com/search/title/?genres={}&sort=num_votes,desc&start={}'.format(genre, str(k) )
		source = get_website_source(url)
		matches = re.finditer('href="/title/(.{,10})/\?ref_=adv_li_i', source)
	
		for match in matches:
			if coll.count_documents( {'movie_id': match.group(1) } ) == 0:
				movies.append( { 'movie_id': match.group(1), 'genre': genre, 'status': 0 } )
			else:
				aux.log('This movie already exists in the database: movie_id = {}.'.format( match.group(1) ))
		k += 50

	coll.insert_many(movies)
	client.close()
	aux.log('Successfully imported {} movies from genre: {}.'.format( len(movies), genre ) )

def check_for_duplicates(collection, field):
	"""
	Checks whether in a given collection there are entries with identical values of the field.
	"""
	client = aux.get_client()
	coll = client['ReviewAnalyser'][collection]
	
	count_by_id = { '$group': { '_id': '$' + field, 'count': { '$sum': 1 } } }
	
	pipeline = [ count_by_id ]

	results = coll.aggregate( pipeline )
	
	duplicate_count = 0

	for r in results:
		if r['count'] > 1:
			aux.log('Duplicates for: {} = {}'.format(field, r['_id']))
			duplicate_count += 1
	
	if duplicate_count == 0:
		aux.log('No duplicates founds.')

	client.close()
	

def get_reviews(movie_id):
	"""
	Given an id of a movie, extracts all the reviews visible on the first page and stores them in the database.
	"""
	url = 'https://www.imdb.com/title/{}/reviews'.format(movie_id)
	source = get_website_source(url)
	reviews = extract_reviews(source)
	
	client = aux.get_client()
	coll = client['ReviewAnalyser']['reviews']
	
	for review in reviews:
		# check if the review is not already in the database to avoid duplicates
		if coll.count_documents( {'movie_id': movie_id, 'content': review['content'] } ) == 0:
			review['movie_id'] = movie_id
			coll.insert_one( review )
		else:
			aux.log('This review already exists in the database: movie_id = {}, content = {}.'.format( movie_id, review['content'] ))

	client.close()
	aux.log('Number of reviews found: {}'.format(str(len(reviews))))

def get_website_source(url):
	with open('sample_headers.json', 'r') as json_file:
		sample_headers = json.load(json_file)
		rng = np.random.default_rng()
		headers = { 'User-Agent': sample_headers[ rng.integers(0, len(sample_headers)) ] }
		r = requests.get(url, headers = headers)
		return r.text

def extract_reviews(source, quality_threshold = 0.5, votes_threshold = 5):
	"""
	Given a source code of a website extracts all the reviews and makes a list of those matching our criteria.
	"""
	matches = re.finditer('class="point-scale">', source)
	reviews = []
	prev_point = 0
	
	matches_list = [ *matches ]
	
	if len(matches_list) > 0:
		for match in matches_list:
			if prev_point > 0:
				review = process_review( aux.sanitise_text( source[ prev_point : match.start() ] ) )
				if review['quality'] >= quality_threshold and review['votes'] >= votes_threshold:
					reviews.append(review)

			prev_point = match.start() - 30
		
		review = process_review( aux.sanitise_text( source[ prev_point : ] ) )

		if review['quality'] >= quality_threshold and review['votes'] >= votes_threshold:
			reviews.append(review)

	return reviews

def process_review(raw_text):
	"""
	Given a raw text of a review, it extracts the relevant information and stores them in a dictionary.
	"""
	review = {}
	# extract the score
	match = re.search('<span>(\d{1,2})</span>', raw_text)
	review['score'] = int( match.group(1) )
	# extract the date
	match = re.search('class="review-date">(.{,20})</span>', raw_text)
	review['date'] = match.group(1)
	# extract the content and how many people found it helpful
	match = re.search('class="text show-more__control">(.+?)</div>.*?([\d,]+) out of ([\d,]+) found this helpful', raw_text)
	review['content'] = match.group(1)
	review['chars'] = len( match.group(1) )
	review['words'] = len( match.group(1).split() )
	# quality is the fraction of people that found the review helpful
	votes = aux.convert_to_int( match.group(3) )
	if votes > 0:
		review['quality'] = np.round( aux.convert_to_int( match.group(2) ) / votes, decimals = 2 )
	else:
		review['quality'] = 0
	# votes is the number of people that assessed the review
	review['votes'] = votes
	return review

def get_all_reviews():
	"""
	Downloads reviews for all the movies in the database whose status is 0 (unprocessed).
	"""
	client = aux.get_client()
	coll = client['ReviewAnalyser']['movies']
	results = coll.find()

	for r in results:
		if r['status'] == 0:
			aux.log('Downloading reviews for movie_id = {}'.format(r['movie_id']))
			get_reviews(r['movie_id'])
			coll.update( { 'movie_id': r['movie_id'] }, { '$set': { 'status': 1 } } )
			time.sleep( aux.get_random_sleep_time() )
	
	aux.log('Downloading finished successfully.')
