# Multi-layer perception for log anomaly detection
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, r2_score
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

        self.y_train_tensor = torch.from_numpy(
            new_y_train_s).type(torch.LongTensor)
        self.y_train_onehot = torch.from_numpy(
            new_y_train_s).type(torch.LongTensor)
        self.x_train_tensor = torch.from_numpy(new_x_train_s).float()

        self.x_test_tensor = torch.from_numpy(self.x_test).float()
        y_test_tensor = torch.from_numpy(self.y_test.reshape(-1, 1))
        self.y_test_onehot = torch.zeros(
            y_test_tensor.shape[0], 2).scatter_(1, y_test_tensor, 1)

    def fit(self):
        print("Training model "+self.name+" ...")
        self.net = Net(self.fea_dim, 200, 2)
        print(self.net)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        loss_func = torch.nn.CrossEntropyLoss()
        counter = 0
        for t in range(self.n):
            out = self.net(self.x_train_tensor)
            loss = loss_func(out, self.y_train_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if percentage(t, self.n) > (counter+1)*self.step:
                counter += 1
                print('model %s itr: %d (%d), loss: %f' %
                      (self.name, t, percentage(t, self.n), loss))
                out_ = self.net(self.x_test_tensor)
                y_ = torch.argmax(out_, -1)
                self.acc = (y_ == torch.from_numpy(
                    self.y_test)).sum()/y_.shape[0]
                self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
                    self.y_test, y_, average='binary')

                self.auc = roc_auc_score(self.y_test, y_)
                print('Testset: Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, acc:{:.3f}\n'.format(
                    self.precision, self.recall, self.f1, self.acc))

    def r2score(self, n_split=10):
        r2_scores = []
        split_size = len(self.x_train)//n_split
        for i in range(n_split):
            # Define the start and end indices for the current split
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_split - \
                1 else len(self.x_train)

            # Split the data
            x_split = self.x_train_tensor[start_idx:end_idx]
            y_split = self.y_train_onehot[start_idx:end_idx]

            # Make predictions on the current split
            y_pred = self.net(x_split)

            y_1 = torch.argmax(y_pred, -1)
            # Calculate the R2 score for the current split
            r2 = r2_score(y_split, y_1)

            # Append the R2 score to the list
            r2_scores.append(r2)
        return(r2_scores)

    def get_metrics(self):
        return [self.precision, self.recall, self.f1, self.acc, self.auc]


# model = MLP_wrapper("m1")
# model.load_data("../logrep/MCV_bglPP-sequential.npz.npz")
# model.train()
# MCV_hdfsPP.npz
#data = "../logrep/MCV_bglPP-sequential.npz.npz"

def run_on(data):

    prec_l = []
    recall_l = []
    f1_l = []
    auc_l = []
    r2 = []
    for i in range(1):
        model = MLP_wrapper(data+" m"+str(i))
        model.load_data(data)
        model.fit()
        r2.append(model.r2score())

        precision, recall, f1, acc, auc = model.get_metrics()
        prec_l.append(precision)
        recall_l.append(recall)
        f1_l.append(f1)
        auc_l.append(auc)

    for i, x in enumerate(r2):
        r2[i] = sum(x)/len(x)

    r2 = sum(r2)/len(r2)

    result_string = f'---------------------------------------\nRUN\nDATA: {data}\n---------------------------------------\nAverage Precision: {sum(prec_l) / len(prec_l)}\nAverage Recall: {sum(recall_l)/len(recall_l)}\nAverage F1: {sum(f1_l) / len(f1_l)}\nAverage AUC: {sum(auc_l) / len(auc_l)} \nAverage R2: {r2}\n---------------------------------------\n\n'

    # Save the result in a file
    with open('./result_MLP.txt', 'a') as file:
        file.write(result_string)

    print(result_string)


run_on("../logrep/MCV_hdfsPP.npz.npz")

# run_on("../logrep/MCV_bglPP-sequential.npz.npz")
