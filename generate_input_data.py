"""
This file is used to generate the input data.
"""

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux

N_reviews = 2000
filename = 'data'
criteria = {'max_words': 250, 'max_sentences': 15, 'max_words_per_sentence': 50, 'quality': 0.7}
	
# generate input data
aux.log('Generate input data from the reviews stored in the database.')
client = aux.get_client()
analyser.generate_input_data(client, filename, N_reviews, criteria)
client.close()
