import torch.nn as nn
import torch.optim as optim
import os

import random
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
import time

from numpy import array
from numpy import hstack

# prints CUDA if we are on CUDA machine

# Hyperparameters
# Data split between trainining and test (percentages)
# note: testsize = 100 - train_sz
train_sz = 80  # currently 80% of the data (before val-data is removed)
val_sz = 20  # currently 20% of training data
n_timesteps = 5  # this is number of timesteps (tau)
learn_rate = 0.01  # Hyperparam
train_episodes = 100  # Hyperparam
batch_size = 16  # how often parameters get updated
num_hidden_states = 200
num_hidden_layers = 3

PATH = "./data/FMCC_1.csv"
SAVE_PATH = "."

with open(PATH, newline='') as csvfile:
    dictreader = csv.DictReader(csvfile, delimiter=',')

    feature_names = dictreader.fieldnames
    all_data = list(dictreader)  # creates a list of dicts (1 per sample) using first row of csv for feature-names
    del all_data[-1]  # delete last item
    del all_data[0]  # delete first item
    print(len(all_data))
    data_length = len(all_data)

    inp_feats = ['Percentage Change (Low)', 'Percentage Change (High)', 'Percentage Change (Close)',
                 'Percentage Change(Volume)', 'Percentage Change(Vix)', 'Percentage Change(S&P)']
    alternate_inp_feats = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'VIX Closed', 'S&P Close']
    outp_feat = 'Next Day Percentage Change (Closed)'

    for i, row in enumerate(all_data):
        try:
            all_data[i][outp_feat] = float(all_data[i][outp_feat][0:-1])
        except ValueError:
            print('Line {} is corrupt!'.format(i))
            break

        for mat_feat in inp_feats:
            all_data[i][mat_feat] = float(all_data[i][mat_feat][0:-1])

    all_inps = np.array([[all_data[samp][feat] for feat in inp_feats] for samp in range(data_length)])
    all_outps = np.array([all_data[samp][outp_feat] for samp in range(data_length)]).reshape(data_length, 1)

    n_features = len(inp_feats)

    dataset = hstack((all_inps, all_outps))

    print(dataset.shape)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


""" LSTM model
"""


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = num_hidden_states  # number of hidden states
        self.n_layers = num_hidden_layers  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)

        # according to pytorch docs LSTM output is
        # (batch_size, seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


# convert dataset into input/output
X, y = split_sequences(dataset, n_timesteps)

# split data into training/test/val
train_test_split = int(train_sz * X.shape[0] // 100)
val_split = (val_sz * train_test_split // 100)

valX = X[:val_split]
valy = y[:val_split]
trainX = X[val_split:train_test_split]
trainy = y[val_split:train_test_split]
testX = X[train_test_split:]
testy = y[train_test_split:]

# create network
mv_net = MV_LSTM(n_features, n_timesteps)
criterion = torch.nn.MSELoss()  # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=learn_rate)

# mv_net.train()
num_epoch = []
loss_list_train = []
loss_list_val = []

loss_order = 10
temp_lr = learn_rate

minimum = 1000

start_time = time.time()
for t in range(train_episodes):
    # train
    mv_net.train()
    for b in range(0, len(trainX), batch_size):
        inpt = trainX[b:b + batch_size, :, :]
        target = trainy[b:b + batch_size]

        x_batch = torch.tensor(inpt, dtype=torch.float32)
        y_batch = torch.tensor(target, dtype=torch.float32)

        mv_net.init_hidden(x_batch.size(0))
        output = mv_net(x_batch)
        loss = criterion(output.view(-1), y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # experiment
        if loss.item() < (loss_order / 100):
            loss_order /= 100
            temp_lr /= 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = temp_lr
            print('cut lr to', temp_lr)

    if t % 20 == 0:
        print('step : ', t, 'loss : ', loss.item())
    num_epoch.append(t)
    loss_list_train.append(loss.item())

    min_error_idx = 0
    # estimate validation error per epoch
    for b in range(0, len(valX), batch_size):
        min_error_idx = b
        inpt = valX[b:b + batch_size, :, :]
        target = valy[b:b + batch_size]

        x_batch_val = torch.tensor(inpt, dtype=torch.float32)
        y_batch_val = torch.tensor(target, dtype=torch.float32)

        mv_net.init_hidden(x_batch_val.size(0))
        output = mv_net(x_batch_val)
        loss_val = criterion(output.view(-1), y_batch_val)

    loss_list_val.append(loss_val.item())

minutes = (time.time() - start_time)
print("--- %s ---" % minutes)

# plot two error curves
d_epoch = np.array(num_epoch)
d_train = np.array(loss_list_train)
d_val = np.array(loss_list_val)
val_min = np.argmin(loss_list_val)
print(val_min)

fig, ax = plt.subplots()
ax.plot(d_epoch, d_train, 'k--', label='train loss')
ax.plot(d_epoch, d_val, 'r--', label='validation loss')
plt.axvline(val_min)
ax.set_title("Training and validation losses")
ax.set_xlabel("Number of epochs")
ax.set_ylabel("Loss")
ax.legend(loc='best')
fig.show()

# test - print out the output
batch_size = 1
num_correct = 0
num_incorrect = 0
mae = []

test_set_targets = []
test_set_outputs = []

for b in range(0, len(testX), batch_size):
    inpt = testX[b:b + batch_size, :, :]
    target = testy[b:b + batch_size]

    x_batch = torch.tensor(inpt, dtype=torch.float32)
    y_batch = torch.tensor(target, dtype=torch.float32)

    mv_net.init_hidden(x_batch.size(0))
    output = mv_net(x_batch)

    # accuracy rate
    abs_diff = np.absolute(output.detach().numpy() - target)
    mae.append(abs_diff)
    # print('ground_truth: ', target, 'output: ', output)

    if np.sign(target) == np.sign(output.detach().numpy()):
        num_correct += 1
    else:
        num_incorrect += 1

    test_set_targets.append(target[0])
    test_set_outputs.append(output.detach().numpy()[0][0])

# confusion matrix
target_tensor = torch.from_numpy(np.array(test_set_targets))
output_tensor = torch.from_numpy(np.array(test_set_outputs))
stacked = torch.stack(
    (target_tensor, output_tensor), dim=1
)


# Trading recommendation
pred_profit = stacked[-1][1]
if pred_profit > 0:
    print("buy")
else:
    print("sell")

cmt = torch.zeros(3, 3, dtype=torch.float64)
target_perf = 1
pred_perf = 1
for p in stacked:
    target, pred = p.tolist()
    actual_perf = 1 + (target / 100)

    if np.sign(target) == -1 and np.sign(pred) == -1:
        cmt[0][0] += 1
        pred_perf *= actual_perf
    elif np.sign(target) == -1 and np.sign(pred) == 0:
        cmt[0][1] += 1
    elif np.sign(target) == -1 and np.sign(pred) == 1:
        cmt[0][2] += 1
        pred_perf *= actual_perf
    elif np.sign(target) == 0 and np.sign(pred) == -1:
        cmt[1][0] += 1
        pred_perf *= actual_perf
    elif np.sign(target) == 0 and np.sign(pred) == 0:
        cmt[1][1] += 1
    elif np.sign(target) == 0 and np.sign(pred) == 1:
        cmt[1][2] += 1
        pred_perf *= actual_perf
    elif np.sign(target) == 1 and np.sign(pred) == -1:
        cmt[2][0] += 1
        pred_perf *= actual_perf
    elif np.sign(target) == 1 and np.sign(pred) == 0:
        cmt[2][1] += 1
    elif np.sign(target) == 1 and np.sign(pred) == 1:
        cmt[2][2] += 1
        pred_perf *= actual_perf

print(cmt)
print("performance based on prediction: ", pred_perf)

mae = np.array(mae)
mae_avg = np.mean(mae)
print('Mean Absolute Error: ', mae_avg)
print('correct:', num_correct)
print('incorrect:', num_incorrect)
print(num_correct / (num_correct + num_incorrect))
print('train size: ', len(trainX))
print('test size: ', len(testX))
