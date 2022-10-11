from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="recommender",
        packages=find_packages(exclude=["recommender_tests"]),
        install_requires=[
            "dagster",
            "scikit-learn",
            "pandas",
            "requests",
        ],
        extras_require={"dev": ["dagit", "pytest"]},
    )
