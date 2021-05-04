import torch
from torch import nn

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

class LSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_layer_size=100, output_size=2, vocab_size=52, embedding_dim):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

  def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions


def evaluate_model(model, data_batches, loss_function):
    eval_loss = 0
    eval_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in data_batches:
            correct = 0
            total = 0
            predictions = model(batch[0]).squeeze(1)
            for j in range(predictions.shape[0]):
                predicted = np.argmax(F.softmax(predictions[j].data))
                true = np.argmax(batch[1][j].data)
                if (predicted == true):
                    correct += 1
                total += 1
            loss = loss_function(predictions, batch[1])
            eval_loss += loss.item()
            eval_acc += (correct / total)
    return eval_loss / len(data_batches), eval_acc / len(data_batches)


if __name__ == '__main__':
    _,testDataLoader, embedding_size = scripts.get_data(data_path="../data/Data1/train_eng.csv",testData = True)
    model = LSTM(input_size = embedding_size, embedding_dim = embedding_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.load_state_dict(torch.load('./model.pt'))
    test_loss, test_acc = evaluate_model(model, testDataLoader, loss_function)
    print(f'Accuracy on test data : {test_acc * 100:.2f}%')
