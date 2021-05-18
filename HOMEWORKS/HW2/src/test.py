from context import scripts
import scripts
import pandas as pd
import numpy as np
import time

df_movie_features, df_movie_features_test, df_ratings, df_ratings_test = scripts.get_data(data_path="../data/Data1/train.csv",testData = True)


def recommend_movies(preds_df, userID, original_ratings_df, num_recommendations=5):
    pred_df = pd.DataFrame(np.array(preds_df), columns=df_movie_features.columns)
    sorted_user_predictions = (pred_df.iloc[userID-1].sort_values(ascending=False))
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    return (pd.DataFrame(sorted_user_predictions.drop(user_data.movieId))[0:num_recommendations] / pd.DataFrame(sorted_user_predictions.drop(user_data.movieId)).max() * 4)


if __name__ == '__main__':
    preds_df = pd.read_pickle('./model')

    print("recomendation for user", pd.unique(df_ratings_test.userId)[280])
    print(recommend_movies(preds_df, pd.unique(df_ratings_test.userId)[280], df_ratings, 20))
