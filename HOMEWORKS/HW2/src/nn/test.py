from context import scripts
import scripts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import collections
import math
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


df_movie_features, df_movie_features_test, df_ratings, df_ratings_test = scripts.get_data(data_path="../data/Data1/train.csv",testData = True)

class RecommenderNet(nn.Module):
  def __init__(self, n_users, n_movies, n_factors=50, embedding_dropout=0.02, dropout_rate=0.2):
    super().__init__()

    self.u = nn.Embedding(n_users, n_factors)
    self.m = nn.Embedding(n_movies, n_factors)
    self.drop = nn.Dropout(embedding_dropout)
    self.hidden1 = nn.Sequential( 
                nn.Linear(100, 128), 
                nn.ReLU(),
                nn.Dropout(dropout_rate)
        )
    self.hidden2 = nn.Sequential( 
                nn.Linear(128, 256), 
                nn.ReLU(),
                nn.Dropout(dropout_rate)
        )
    self.hidden3 = nn.Sequential( 
                nn.Linear(256, 128), 
                nn.ReLU(),
                nn.Dropout(dropout_rate)
        )
    self.fc = nn.Linear(128, 1)
    self._init()

  def forward(self, users, movies, minmax=[1,4]):
    features = torch.cat([self.u(users), self.m(movies)], dim=1)
    x = self.drop(features)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.hidden3(x)
    out = torch.sigmoid(self.fc(x))
    
    if minmax is not None:
      min_rating, max_rating = minmax
      out = out*(max_rating - min_rating + 1) + min_rating - 0.5
    return out

  def _init(self):
    def init(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    self.u.weight.data.uniform_(-0.05, 0.05)
    self.m.weight.data.uniform_(-0.05, 0.05)
    self.hidden1.apply(init)
    init(self.fc)

def predict(model, userId, batches):
    ratings = dict()
    for user_batch, movie_batch, rate_batch in batches:
      out = model(user_batch.to(device)[0:user_batch.shape[0]-1], movie_batch.to(device)[0:movie_batch.shape[0]-1])
      rats = (np.array(out.detach().numpy()))
      for i in range(rats.shape[0]):
        ratings.setdefault(rats[i][0], 0)
        ratings[rats[i][0]] = movie_batch[i].item()
    return collections.OrderedDict(sorted(ratings.items(), reverse=True))

def recommend_movies_nn(model, userId, recommend_num = 10):
    batches_to_predict = []
    unwatched_movies = np.unique(df_ratings_test[df_ratings_test["userId"]!=userId]["movieId"])
    users_array = np.full((unwatched_movies.shape), userId)
    batches_to_predict.append((torch.tensor(users_array, dtype=torch.long), torch.tensor(unwatched_movies, dtype=torch.long), None))
    ratings = predict(model, userId, batches_to_predict)
    array_keys = np.array(ratings.keys())
    res = dict(zip(ratings.values(), [math.floor(x) for x in ratings.keys()]))
    return pd.DataFrame.from_dict(res, orient='index', columns=["Ratings"])[0:recommend_num]


if __name__ == '__main__':
    users = df_ratings['userId'].values - 1
    movies = df_ratings['movieId'].values - 1
    n_users, n_movies =  max(users)+1, max(movies)+1
    model = RecommenderNet(n_users=n_users, n_movies=n_movies).to(device)
    model.load_state_dict(torch.load('./weights.h5'))
    model.eval()

    print("recomendation for user", df_ratings_test.userId[7000])
    print(recommend_movies_nn(net, df_ratings_test.userId[7000], 20))
