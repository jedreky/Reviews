"""
This file contains some simple procedures for mass-downloading reviews.
"""

import reviews.crawler as crawler

#n = 10
#genres = ('action', 'adventure', 'thriller')

#for genre in genres:
#	crawler.get_movies_from_genre(genre, n)

crawler.get_all_reviews()
