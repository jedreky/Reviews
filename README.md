# Reviews

The goal of this project is to build and train a machine learning (ML) model capable of performing sentiment analysis of movie reviews.

The first module (crawler.py) provides functions necessary to download a large number of movie reviews from IMDB.com and store them in a local Mongo database. For each review we save the content of the review, the score (on the scale from 1 to 10) and the quality (assessed by the number of people that had found them helpful).

The second module (analyser.py) is responsible for the ML part of the project. We first filter the reviews stored in the database according to some criteria in terms of length, quality and then we pick a sample which is balanced in terms of the given score. We then convert the reviews into numerical data by replacing every words by its pretrained vector representation (we use the word embeddings provided by the GloVe project: https://nlp.stanford.edu/projects/glove/). We then construct a recurrent neural network (RNN) of specified parameters and train in on the movie reviews. Finally, we evaluate the performance of the model on previously unseen reviews and visualise the results.
