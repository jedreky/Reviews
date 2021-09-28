"""
This files takes data from the database and generates an appropriate .npz file.
"""

import reviews.analyser as analyser

filename = 'data'
n = 20
max_words = 150
quality = 0.5

analyser.generate_input_data(filename, n, max_words, quality)
