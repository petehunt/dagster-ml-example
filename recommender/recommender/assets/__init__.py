from dagster import asset
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher
from collections import Counter
from types import SimpleNamespace
from sklearn.metrics import pairwise_distances
import numpy as np


@asset(config_schema={"small": bool})
def movielens_zip(context):
    small = context.op_config["small"]
    if small:
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    else:
        url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    return requests.get(url).content


@asset
def movielens_ratings(movielens_zip):
    with ZipFile(BytesIO(movielens_zip)) as archive:
        filename = [filename for filename in archive.namelist(
        ) if filename.endswith("ratings.csv")][0]
        return pd.read_csv(archive.open(filename))


@asset
def movielens_movies(movielens_zip):
    with ZipFile(BytesIO(movielens_zip)) as archive:
        filename = [filename for filename in archive.namelist(
        ) if filename.endswith("movies.csv")][0]
        return pd.read_csv(archive.open(filename))


@asset
def movie_to_users(movielens_ratings):
    # create a matrix: 1 row per movie, 1 column per user
    df = movielens_ratings[["movieId", "userId"]].groupby(
        "movieId").aggregate(set).reset_index()
    movie_ids = list(df["movieId"])
    fh = FeatureHasher()
    features = fh.fit_transform(
        [Counter(str(user_id) for user_id in user_ids) for user_ids in df["userId"]])

    return SimpleNamespace(movie_ids=movie_ids, features=features)


class RecommenderModel:
    def __init__(self, features, ids):
        # train the model by eagerly computing the distances between
        # every pair of movies. there are better ways of doing this,
        # including using an approximate nearest neighbors index
        # such as faiss, however, this is sufficient for the purpose
        # of this demo.
        self.matrix = pairwise_distances(
            features, features, metric="cosine", n_jobs=-1)
        self.ids = ids

    def find_similar(self, id, n=5):
        index = self.ids.index(id)
        row = self.matrix[index]
        top_indexes = np.argsort(row)
        return [self.ids[index] for index in top_indexes[:n]]


@asset
def movie_recommender_model(movie_to_users):
    return RecommenderModel(movie_to_users.features, movie_to_users.movie_ids)
