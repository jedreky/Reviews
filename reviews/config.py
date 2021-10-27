"""
This file sets some global variables and parameters.
"""

# name of the Mongo database
database_name = 'Reviews'

# path to the embedding dictionary files
emb_dict_file = 'glove.6B/glove.6B.{}d'

# valid embedding dimensions
#emb_dims = ( 50, 100, 200, 300 )
emb_dims = ( 50, 100, 200 )

# The maximal length of a review
# DO WE STILL NEED THIS?
max_words = 150

# number of movies on every page
movies_per_page = 50

# number of epochs between accuracy checks
N_epochs = 10

# number of seconds in an hour
secs_in_hr = 3600
