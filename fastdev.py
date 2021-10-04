"""
This file is used for fast development of simple features.
"""

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

max_words = 150
quality = 0.75

analyser.check_score_distribution(max_words, quality)
