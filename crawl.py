"""
This file contains some simple procedures for downloading and processing reviews.
"""

import reviews.auxiliary_functions as aux
import reviews.crawler as crawler

#n = 10
#genres = ('action', 'adventure', 'thriller')

client = aux.get_client()

#for genre in genres:
#	crawler.get_movies_from_genre(client, genre, n)

#crawler.get_all_reviews(client)
crawler.process_raw_reviews(client)

client.close()
