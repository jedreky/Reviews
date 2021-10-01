"""
This file sets some global variables and parameters.
"""

# name of the Mongo database
database_name = 'Reviews'

# The maximal length of a review
max_words = 150

# number of movies on every page
movies_per_page = 50

# valid embedding dimensions
emb_dims = ( 50, 100, 200, 300 )

# number of epochs between accuracy checks
N_epochs = 10

# numerical value of accuracy ratio when the initial accuracy equals 0
# TODO: most likely not needed anymore
max_accuracy = 100

# number of seconds in an hour
secs_in_hr = 3600
