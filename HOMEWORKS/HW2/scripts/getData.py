import numpy as np
import math
import pandas as pd
import os


def get_data(data_path="../data/Data1/train.csv",testData = True):
    assert os.path.isfile(data_path), f"{os.path.realpath(data_path)} : File not exist"

    df_ratings = pd.read_csv(data_path, engine='python' ,encoding = "latin-1", usecols=["userId","movieId","rating"], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    df_ratings_test = pd.read_csv("../data/Data1/test.csv", header=0, engine='python' ,encoding = "latin-1", usecols=["userId","movieId","rating"], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    
    df_movie_features = df_ratings.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    df_movie_features_test = df_ratings_test.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    return df_movie_features, df_movie_features_test, df_ratings, df_ratings_test



if __name__ == '__main__':
    pass
    # _,_ = get_data(testData=True)
