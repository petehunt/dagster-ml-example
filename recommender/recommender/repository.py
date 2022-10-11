from dagster import load_assets_from_package_module, repository

from recommender import assets


@repository
def recommender():
    return [load_assets_from_package_module(assets)]
