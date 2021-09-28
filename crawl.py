"""
This file contains some simple procedures for downloading reviews.
"""

import reviewanalyser.crawler as crawler

n = 10000
genres = ('action', 'adventure', 'thriller')

#for genre in genres:
#	crawler.get_movies_from_genre(genre, n)

crawler.get_all_reviews()
