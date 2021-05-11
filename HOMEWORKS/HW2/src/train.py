from context import scripts
import scripts
import pandas as pd
import numpy as np
import time
import math

df_movie_features, _, df_ratings, df_ratings_test = scripts.get_data(data_path="../data/Data1/train.csv",testData = True)
K=50
alpha=0.01
beta = 0.001
R = np.array(df_movie_features)
latent_dim = K
iterations = 20
num_users, num_items = R.shape
P=np.random.normal(scale=1./latent_dim, size=(num_users, latent_dim))
Q=np.random.normal(scale=1./latent_dim, size=(num_items, latent_dim))
b_u = np.zeros(num_users)
b_i = np.zeros(num_items)
b = np.mean(R[np.where(R != 0)])
samples = [
    (i, j, R[i, j])
    for i in range(num_users)
    for j in range(num_items)
    if R[i, j] > 0
]

def train():
    for i in range(iterations):
        np.random.shuffle(samples)
        sgd()
        losss = mse()
        #print("Iteration = ", i, ", losses=", losss)

def mse():
    xs, ys = R.nonzero()
    predicted = full_matrix()
    error = 0
    counter=0
    for x, y in zip(xs, ys):
        error += pow(R[x, y] - predicted[x, y], 2)
        counter+=1
    return np.sqrt(error/counter)

def sgd():
    for i, j, r in samples:
        prediction = get_rating(i, j)
        if math.isnan(prediction):
          get_rating_with_print(i,j)
          break
        e = (r - prediction)
        b_u[i] += alpha * (e - beta * b_u[i])
        b_i[j] += alpha * (e - beta * b_i[j])
        P[i, :] += alpha * (e * Q[j, :] - beta * P[i,:])
        Q[j, :] += alpha * (e * P[i, :] - beta * Q[j,:])

def get_rating(i, j):
    return b + b_u[i] + b_i[j] + P[i, :].dot(Q[j, :].T)

def full_matrix():
    return b + b_u[:,np.newaxis] + b_i[np.newaxis:,] + P.dot(Q.T)


def main(args):
    train()
    preds_df = pd.DataFrame(full_matrix())

    preds_df.to_pickle("model")

if __name__ == '__main__':
    main(None)
