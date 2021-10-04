"""
This file is used for fast development of simple features.
"""

import reviews.analyser as analyser
import reviews.auxiliary_functions as aux
import reviews.config as config

quality = 0.75
max_words = 150

analyser.check_score_distribution(max_words, quality)
