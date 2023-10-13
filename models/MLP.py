# Multi-layer perception for log anomaly detection
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
print("Loading packages...")
print(torch.__version__)
print("done")


def percentage(t, n):
    return round(t/n*100, 2)


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.output(out)
        return out


class MLP_wrapper():

    def __init__(self, name, n=2000, step=10):
        self.n = n
        self.name = name
        self.step = step

    def load_data(self, file):
        print("Loading data "+file+" ...")
        seq_level_data = np.load(file, allow_pickle=True)
        self.x_train = seq_level_data["x_train"]
        self.y_train = seq_level_data["y_train"]
        self.x_test = seq_level_data["x_test"]
        self.y_test = seq_level_data["y_test"]
        print("done")
        self.x_train = np.array(self.x_train, dtype=np.float64)
        self.x_test = np.array(self.x_test, dtype=np.float64)
        pos = self.y_train == 1
        print(self.x_train.shape)
        print(self.y_train.shape)
        print('positive:', sum(pos))
        print('ratio:', self.x_train.shape[0]/sum(pos))
        ratio = self.x_train.shape[0]/sum(pos)
        index = np.argwhere(pos == True)
        self.fea_dim = self.x_train.shape[1]

        pos_items = self.x_train[index, :].reshape(-1, self.fea_dim)
        pos_rep = np.repeat(pos_items, ratio-1, axis=0)

        new_x_train = np.concatenate((self.x_train, pos_rep), axis=0)
        new_y_train = np.append(self.y_train, np.ones(pos_rep.shape[0]))
        index = np.arange(new_x_train.shape[0])
        np.random.shuffle(index)

        new_x_train_s = new_x_train[index]
        new_y_train_s = new_y_train[index]
        batch_size = self.y_train.shape[0]

        y_train_tensor = torch.from_numpy(new_y_train_s).type(torch.LongTensor)
        self.y_train_onehot = torch.from_numpy(
            new_y_train_s).type(torch.LongTensor)
        self.x_train_tensor = torch.from_numpy(new_x_train_s).float()

        self.x_test_tensor = torch.from_numpy(self.x_test).float()
        y_test_tensor = torch.from_numpy(self.y_test.reshape(-1, 1))
        self.y_test_onehot = torch.zeros(
            y_test_tensor.shape[0], 2).scatter_(1, y_test_tensor, 1)

    def train(self):
        print("Training model "+self.name+" ...")
        net = Net(self.fea_dim, 200, 2)
        print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = torch.nn.CrossEntropyLoss()
        counter = 0
        for t in range(self.n):
            out = net(self.x_train_tensor)
            loss = loss_func(out, self.y_train_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if percentage(t, self.n) > (counter+1)*self.step:
                counter += 1
                print('model %s itr: %d (%d), loss: %f' %
                      (self.name, t, percentage(t, self.n), loss))
                out_ = net(self.x_test_tensor)
                y_ = torch.argmax(out_, -1)
                self.acc = (y_ == torch.from_numpy(
                    self.y_test)).sum()/y_.shape[0]
                self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
                    self.y_test, y_, average='binary')
                print('Testset: Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, acc:{:.3f}\n'.format(
                    self.precision, self.recall, self.f1, self.acc))

    def get_metrics(self):
        return [self.precision, self.recall, self.f1, self.acc]


model = MLP_wrapper("m1")
model.load_data("../logrep/MCV_bglPP-sequential.npz.npz")
model.train()
