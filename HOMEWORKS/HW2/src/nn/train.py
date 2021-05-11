from context import scripts
import scripts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_sz = 128

df_movie_features, _, df_ratings, df_ratings_test = scripts.get_data(data_path="../data/Data1/train.csv",testData = True)


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

def main(args):

    users = df_ratings['userId'].values - 1
    movies = df_ratings['movieId'].values - 1
    rates = df_ratings['rating'].values
    n_samples = len(rates)
    n_users, n_movies =  max(users)+1, max(movies)+1
    batches = []

    for i in range(0, n_samples, batch_sz):
      limit =  min(i + batch_sz, n_samples)
      users_batch, movies_batch, rates_batch = users[i: limit], movies[i: limit], rates[i: limit]
      batches.append((torch.tensor(users_batch, dtype=torch.long), torch.tensor(movies_batch, dtype=torch.long),
                      torch.tensor(rates_batch, dtype=torch.float)))
    users = None
    movies = None 
    rates = None 

    net = RecommenderNet(n_users=n_users, n_movies=n_movies).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2)

    epochs = 10

    for epoch in range(epochs):
      train_loss = 0
      for users_batch, movies_batch, rates_batch in batches:
        net.zero_grad()
        out = net(users_batch.to(device), movies_batch.to(device), [1, 4]).squeeze()
        loss = criterion(rates_batch.to(device), out)

        loss.backward()
        optimizer.step()
        train_loss += loss
      scheduler.step(loss)
      print("Loss at epoch {} = {}".format(epoch, loss.item()))
    print("Last Loss = {}".format(loss.item())) 


    torch.save(net.state_dict(), 'model.pt')

if __name__ == '__main__':
    main(None)
