"""
This file contains functions related to crawling the IMDB website, extracting reviews and storing them in a Mongo database.
"""

import json
import numpy as np
import re
import requests
import time

import reviews.auxiliary_functions as aux
import reviews.config as config

def get_movies_from_genre(client, genre, n):
	"""
	Finds at least n new movies from the given genre and adds them to the database.
	"""
	# create an empty list of movies to add
	movies = []

	coll = client[config.database_name]['movies']

	k = 1
	while n > len(movies):
		url = 'https://www.imdb.com/search/title/?genres={}&sort=num_votes,desc&start={}'.format(genre, str(k) )
		source = get_website_source(url)
		matches = re.finditer('href="/title/(.{,10})/\?ref_=adv_li_i', source)
	
		for match in matches:
			movie_id = match.group(1)
			
			# check if the movie already appears in the database
			if coll.count_documents( {'movie_id': movie_id } ) == 0:
				# create a simple dictionary and add to the list
				movies.append( { 'movie_id': movie_id, 'genre': genre, 'status': 0 } )
			else:
				aux.log('This movie already exists in the database: movie_id = {}.'.format( match.group(1) ))
		# increase the counter by the number of movies on each page
		k += config.movies_per_page

	coll.insert_many(movies)
	aux.log('Successfully imported {} movies from genre: {}.'.format( len(movies), genre ) )

def get_reviews(client, movie_id):
	"""
	Given an id of a movie, extracts all the reviews visible on the first page and stores them in the database.
	"""
	url = 'https://www.imdb.com/title/{}/reviews'.format(movie_id)
	source = get_website_source(url)
	reviews = extract_reviews(source)
	
	coll = client[config.database_name]['raw_reviews']
	
	for review in reviews:
		# check if the review is not already in the database to avoid duplicates
		if coll.count_documents( {'movie_id': movie_id, 'content': review['content'] } ) == 0:
			review['movie_id'] = movie_id
			coll.insert_one( review )
		else:
			aux.log('This review already exists in the database: movie_id = {}, content = {}.'.format( movie_id, review['content'] ))

	aux.log('Number of reviews found: {}'.format(str(len(reviews))))

def get_website_source(url):
	with open('json_data/sample_headers.json', 'r') as json_file:
		sample_headers = json.load(json_file)
		rng = np.random.default_rng()
		headers = { 'User-Agent': sample_headers[ rng.integers(0, len(sample_headers)) ] }
		r = requests.get(url, headers = headers)
		return r.text

def extract_reviews(source):
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
				reviews.append(review)

			prev_point = match.start() - 30
		
		review = process_review( aux.sanitise_text( source[ prev_point : ] ) )

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

def get_all_reviews(client):
	"""
	Downloads reviews for all the movies in the database whose status is 0 (unprocessed).
	Recall that we only download reviews visible on the first page.
	"""
	coll = client[config.database_name]['movies']
	# find movies which have not been processed yet
	results = coll.find( {'status': 0} )

	for r in results:
		movie_id = r['movie_id']
		aux.log('Downloading reviews for movie_id = {}'.format(movie_id))
		get_reviews(movie_id)
		coll.update( { 'movie_id': movie_id }, { '$set': { 'status': 1 } } )
		#time.sleep( aux.get_random_sleep_time() )
	
	aux.log('Downloading finished successfully.')

def sanitise_review_content(content):
	"""
	Sanitises the content of a review by removing non-character symbols,
	removing some abbreviations and unifying all end-of-sentence characters.
	"""
	content = content.lower()

	# expand shortenings
	content = content.replace("won't", 'will not')
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
