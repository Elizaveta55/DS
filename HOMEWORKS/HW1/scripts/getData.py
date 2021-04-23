import torch, os
import torch
from torch import nn
import torchtext
from torchtext.vocab import Vocab
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torchtext.data.utils import get_tokenizer
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_process(raw_text_iter,max_len,tokenizer, vocab_new):
    batch = []
    for item in raw_text_iter:
      res = []
      for i in range(max_len):
        if (len(item)>i):
          res.extend([vocab_new[token] for token in tokenizer(item[i])])
        else:
          res.extend([0])
      batch.append(res)
    pad_data = torch.FloatTensor(batch)
    return pad_data

def get_data(data_path="../data/Data1/train_eng.csv",testData = True):
    assert os.path.isfile(data_path), f"{os.path.realpath(data_path)} : File not exist"

    train_data = pd.read_csv(data_path, header=0, engine='python' ,encoding = "latin-1", usecols=["Name","Gender"])
    test_data = pd.read_csv("../data/Data1/test_eng.csv", header=0, engine='python' ,encoding = "latin-1", usecols=["Name","Gender"])
    test_data['Gender'] = test_data['Gender'].apply(lambda x: 0 if x=='M' else 1)
    train_data['Gender'] = train_data['Gender'].apply(lambda x: 0 if x=='M' else 1)
    train_data = train_data.sort_values(by="Name", key=lambda x: x.str.len())
    test_data = test_data.sort_values(by="Name", key=lambda x: x.str.len())
    max_length_test = len(test_data.iloc[-1]['Name'])
    max_length_train = len(train_data.iloc[-1]['Name'])

    unique = list(set("".join(train_data.iloc[:,0])))
    unique.sort()
    vocab = dict(zip(unique, range(1,len(unique)+1)))
    tokenizer = get_tokenizer('basic_english')
    vocab_new = Vocab(vocab,specials=())
    scaler = MinMaxScaler(feature_range=(-1, 1))

    max_len = 64
    embedding_size = max(max_length_train, max_length_test)
    n_classes = len(np.unique(train_data.Gender.values))

    test_tensor = data_process(test_data.Name.values, embedding_size, tokenizer, vocab_new)
    test_data_normalized = torch.FloatTensor(scaler.fit_transform(test_tensor))
    test_tgts_tensor = torch.nn.functional.one_hot(torch.from_numpy(test_data.Gender.values), n_classes) #torch.from_numpy(train_data.Target.values)
    test_dataset = TensorDataset(test_data_normalized, test_tgts_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, pin_memory=True)

    train_tensor = data_process(train_data.Name.values, embedding_size, tokenizer, vocab_new)
    train_data_normalized = torch.FloatTensor(scaler.fit_transform(train_tensor))
    tgts_tensor = torch.nn.functional.one_hot(torch.from_numpy(train_data.Gender.values), n_classes) #torch.from_numpy(train_data.Target.values)
    dataset = TensorDataset(train_data_normalized, tgts_tensor)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    return train_loader, test_loader, embedding_size



if __name__ == '__main__':
    pass
    # _,_ = get_data(testData=True)
