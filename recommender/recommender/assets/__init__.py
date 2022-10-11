from dagster import asset
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher
from collections import Counter
from types import SimpleNamespace
from sklearn.neighbors import NearestNeighbors


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


@asset
def movie_recommender_model(movie_to_users):
    model = NearestNeighbors(n_jobs=-1)
    model.fit(movie_to_users.features)
    return SimpleNamespace(movie_ids=movie_to_users.movie_ids, model=model)
