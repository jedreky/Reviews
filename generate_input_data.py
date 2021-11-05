"""
This file is used to generate the input data.
"""
# Reviews imports
import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
#######################################################
# Generate input data
#######################################################
N_reviews = 950
filename = 'data'
criteria = {'max_words': 250, 'max_sentences': 15, 'max_words_per_sentence': 50, 'quality': 0}

client = aux.get_client()
# check the distribution of scores for given criteria
analyser.check_score_distribution( client, criteria )
# generate input data
aux.log('Generate input data from the reviews stored in the database.')
analyser.generate_input_data(client, filename, N_reviews, criteria)
client.close()
