from recommender.assets import movielens_zip, movielens_ratings, movielens_movies, movie_to_users, movie_recommender_model
import pandas as pd
import os
from dagster import build_op_context


def test_smoke():
    # NOTE: if you don't want to hit the network from a test,
    # copy the zip file into your repo and use the code below.
    # We don't do that for this example since we are not allowed
    # to reproduce the MovieLens dataset.

    # File available at: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
    # zip_file = open(os.path.join(os.path.dirname(
    #    __file__), 'ml-latest-small.zip'), "rb").read()

    zip_file = movielens_zip(build_op_context(config={"small": True}))

    movies = movielens_movies(zip_file)
    users = movie_to_users(movielens_ratings(zip_file))
    result = movie_recommender_model(users)

    def movie_id_to_index(movie_id):
        return result.movie_ids.index(movie_id)

    def index_to_movie_id(index):
        return result.movie_ids[index]

    def movie_id_to_title(movie_id):
        return movies.loc[movies["movieId"] == movie_id]["title"].iloc[0]

    def get_similar_movie_titles(movie_id):
        similar_movie_ids = result.model.kneighbors(
            users.features[[movie_id_to_index(movie_id)]], return_distance=False)[0]
        return [movie_id_to_title(index_to_movie_id(index))
                for index in similar_movie_ids]

    die_hard_2_movie_id = 1370
    assert movie_id_to_title(die_hard_2_movie_id) == "Die Hard 2 (1990)"

    cinderella_movie_id = 1022
    assert movie_id_to_title(cinderella_movie_id) == "Cinderella (1950)"

    # these results seem reasonable by human eval
    assert get_similar_movie_titles(die_hard_2_movie_id) == [
        'Die Hard 2 (1990)',
        'Peacemaker, The (1997)',
        'First Blood (Rambo: First Blood) (1982)',
        'Rambo: First Blood Part II (1985)',
        'For Your Eyes Only (1981)'
    ]

    assert get_similar_movie_titles(cinderella_movie_id) == [
        'Cinderella (1950)',
        'Sleeping Beauty (1959)',
        'Peter Pan (1953)',
        'Alice in Wonderland (1951)',
        'Mouse Hunt (1997)'
    ]
