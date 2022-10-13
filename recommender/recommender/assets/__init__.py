from sklearn.decomposition import TruncatedSVD
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


def extract_file_from_zip(zip_file_bytes, filename):
    with ZipFile(BytesIO(zip_file_bytes)) as archive:
        full_path = [candidate_path for candidate_path in archive.namelist(
        ) if candidate_path.endswith(filename)][0]
        return pd.read_csv(archive.open(full_path))


@asset
def movielens_ratings(movielens_zip):
    return extract_file_from_zip(movielens_zip, "ratings.csv")


@asset
def movielens_movies(movielens_zip):
    return extract_file_from_zip(movielens_zip, "movies.csv")


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


@asset
def movie_to_users_compressed(movie_to_users):
    return SimpleNamespace(
        movie_ids=movie_to_users.movie_ids,
        features=TruncatedSVD(100, random_state=42).fit_transform(
            movie_to_users.features)
    )


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
def movie_recommender_model(movie_to_users_compressed):
    return RecommenderModel(movie_to_users_compressed.features, movie_to_users_compressed.movie_ids)
