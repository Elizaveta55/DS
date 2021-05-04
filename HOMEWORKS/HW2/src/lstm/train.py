from context import scripts
import scripts
import torch
from torch import nn
import torchtext
import time
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu' # get device for training model
writer = SummaryWriter()

class LSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

  def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    all = y_pred.shape[0]
    sum = 0
    for i in range(all):
        sum += f1_loss_one(y_true[i], y_pred[i])
    return sum / all


def f1_loss_one(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    y_pred_soft = (F.softmax(((y_pred)), dim=0)).detach()
    if y_pred_soft.ndim == 2:
        y_pred_soft = y_pred_soft.argmax(dim=1)
    if y_true.ndim == 2:
        y_true = y_true.argmax(dim=1)

    tp = (y_true * y_pred_soft).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred_soft)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred_soft).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred_soft)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1

def main(args):

    #load data
    train_DataLoader, _, embedding_size  = scripts.get_data(data_path="../data/Data1/train_eng.csv",testData = True)
    # train_DataLoader = DataLoader(TensorDataset(train_x, train_y), batch_size=30)

    model = LSTM(input_size = embedding_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter()

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard

    # Commented out IPython magic to ensure Python compatibility.
    # %tensorboard --logdir runs

    epochs = 50
    start_time = time.time()

    for i in range(epochs):
        correct = 0
        total = 0
        start_epoch = time.time()
        for item in train_DataLoader:
            seq = item[0]
            label = item[1]
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)
            for j in range(y_pred.shape[0]):
                predicted = np.argmax(F.softmax(y_pred[j].data))
                true = np.argmax(label[j].data)
                if (predicted == true):
                    correct += 1
                total += 1
            single_loss = loss_function(y_pred, label.float())
            single_loss.backward()
            optimizer.step()

        time_per_epoch = time.time() - start_epoch
        f_measure = f1_loss(label, y_pred)
        writer.add_scalar("train_loss", single_loss.item(), i)
        writer.add_scalar("train_acc", (100 * correct / total), i)
        writer.add_scalar("train_measure", f_measure, i)
        writer.add_scalar("train_time", time_per_epoch, i)
        for tag, parm in model.named_parameters():
            writer.add_histogram(tag, parm.grad.data.cpu().numpy(), i)
        print(
            f'epoch: {i:3} loss: {single_loss.item():10.8f}, accuracy: {(100 * correct / total)}, f-measure: {f_measure}, time = {time_per_epoch}')

    print("MODEL TIME EXECUTION--- %s seconds ---" % (time.time() - start_time))
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main(None)
