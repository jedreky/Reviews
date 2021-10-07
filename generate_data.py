"""
This files takes data from the database and generates an appropriate .npz file.
"""

import reviews.analyser as analyser
import reviews.config as config

filename = 'data'
n = 1000
quality = 0.75

for padding in ('pre', 'post'):
	analyser.generate_input_data(filename, n, config.max_words, quality, padding)
