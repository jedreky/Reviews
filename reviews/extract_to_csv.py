import csv
import json
import pymongo


def get_client():
    """
	Returns a MongoClient.
	"""
    client = None

    with open("json_data/mongo_keys.json", "r") as json_file:
        mongo_keys = json.load(json_file)
        client = pymongo.MongoClient(
            username=mongo_keys["user"], password=mongo_keys["password"]
        )

    return client


def generate_filter(criteria):
    filt = []

    # add max_words criterion
    if "max_words" in criteria:
        filt.append({"$match": {"words": {"$lte": criteria["max_words"]}}})

    # add max_sentences criterion
    if "max_sentences" in criteria:
        filt.append({"$match": {"sentences": {"$lte": criteria["max_sentences"]}}})

    # add max_words_per_sentence criterion
    if "max_words_per_sentence" in criteria:
        filt.append(
            {
                "$match": {
                    "words_per_sentence": {"$lte": criteria["max_words_per_sentence"]}
                }
            }
        )

    # add votes criterion
    if "votes" in criteria:
        filt.append({"$match": {"quality": {"$gte": criteria["votes"]}}})

    # add quality criterion
    if "quality" in criteria:
        filt.append({"$match": {"quality": {"$gte": criteria["quality"]}}})

    return filt


def extract_reviews_to_csv(client, filename, N_reviews):
    coll = client["Reviews"]["reviews"]

    criteria = {"max_words": 300, "quality": 0.8, "votes": 10}
    filt = generate_filter(criteria)

    csv_data = []

    # iterate over all the scores
    for score in range(1, 11):
        score_criterion = [{"$match": {"score": score}}]
        results = coll.aggregate(filt + score_criterion + [{"$count": "count"}])

        # check if a sufficient number of reviews is found for each score
        # TODO: take care of empty result set
        if results.next()["count"] >= N_reviews:
            results = coll.aggregate(filt + score_criterion)

            for j in range(N_reviews):
                r = results.next()
                csv_data.append([r["content"], r["score"]])
        else:
            raise RuntimeError("Not enough reviews")

    # csv_data = [["asf",1], ["sadffdsa", 5], ["sadf", 3], ["f", 8]]
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["content", "score"])
        for row in csv_data:
            writer.writerow(row)

if __name__ == "__main__":
    client = get_client()
    extract_reviews_to_csv(client, "reviews.csv", 10)
