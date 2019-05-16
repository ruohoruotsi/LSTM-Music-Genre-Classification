#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Keras implementation of a simple 2-layer-deep LSTM for genre classification of musical audio.
    Feeding the LSTM stack are spectral {centroid, contrast}, chromagram & MFCC features (33 total values)

    Question: Why is there a pytorch implementation, when we already have Keras/Tensorflow?
    Answer:   So that we can learn more PyTorch on an easy problem! I'm am also curious
              about the performances of both toolkits.

    The plan, first start with a torch.nn implementation, then go for the torch.nn.LSTMCell

"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)


batch_size = 35  # num of training examples per minibatch
num_epochs = 400

genre_features = GenreFeatureData()

# if all of the preprocessed files do not exist, regenerate them all for self-consistency
if (
    os.path.isfile(genre_features.train_X_preprocessed_data)
    and os.path.isfile(genre_features.train_Y_preprocessed_data)
    and os.path.isfile(genre_features.dev_X_preprocessed_data)
    and os.path.isfile(genre_features.dev_Y_preprocessed_data)
    and os.path.isfile(genre_features.test_X_preprocessed_data)
    and os.path.isfile(genre_features.test_Y_preprocessed_data)
):
    print("Preprocessed files exist, deserializing npy files")
    genre_features.load_deserialize_data()
else:
    print("Preprocessing raw audio files")
    genre_features.load_preprocess_data()

# Convert {training, test} torch.Tensors
print(genre_features.train_X.shape)
print(genre_features.test_X.shape)
print(genre_features.train_Y.shape)
print(genre_features.test_Y.shape)

X_train = torch.from_numpy(genre_features.train_X).type(torch.Tensor)
X_test = torch.from_numpy(genre_features.test_X).type(torch.Tensor)
y_train = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor)   # Target is a long tensor of size (N,) which tells the true class of the sample.
y_test = torch.from_numpy(genre_features.test_Y).type(torch.LongTensor)     # Target is a long tensor of size (N,) which tells the true class of the sample.

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

# class definition here
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, hidden = self.lstm(input)

        # Only take the output from the final timetep
        # input_to_linear = lstm_out[-1].view(self.batch_size, -1)
        input_to_linear = lstm_out[-1]  # grab last element of sequence to stuff into linear

        # print(input_to_linear.shape)
        y_pred = self.linear(input_to_linear)
        # print(y_pred.view(-1).shape)
        # return y_pred.view(-1)

        genre_scores = F.log_softmax(y_pred, dim=1)
        return genre_scores

lstm_input_size = 33
hidden_layer_size = 128
output_dimension = 8
num_layers = 2
learning_rate = 0.001

num_batches = int(X_train.shape[0] / batch_size)    # all training data (epoch) / batch_size == num_batches (12)

# Define model
print("Build LSTM RNN model ...")
model = LSTM(lstm_input_size, hidden_layer_size, batch_size=batch_size,
             output_dim=output_dimension, num_layers=num_layers)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
print("Training ...")

hist = np.zeros(num_epochs)

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    # Init hidden state - if you don't want a stateful LSTM (between epochs)
    model.hidden = model.init_hidden()
    for i in range(num_batches):  # for all 420 sequences

        # zero out gradient, so they don't accumulate btw epochs
        model.zero_grad()

        # Local batches and labels
        X_local, y_local = X_train[i * batch_size:(i + 1) * batch_size, ], \
                           y_train[i * batch_size:(i + 1) * batch_size, ]

        # massage input & targets to "match" what the loss_function wants
        X_local = X_local.permute(1, 0, 2)
        # print("y_local shape, before:")
        # print(y_local.shape)
        y_local = torch.max(y_local, 1)[1]  # NLLLoss does not expect a one-hot encoded vector as the target, but class indices

        # print("y_local shape, after:")
        # print(y_local.shape)

        y_pred = model(X_local)                 # fwd the bass (forward pass)
        loss = loss_function(y_pred, y_local)   # compute loss
        loss.backward()                         # reeeeewind (backward pass)
        optimizer.step()                        # parameter update

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(y_pred, y_local, model.batch_size)


    #if epoch % 100 == 0:
    print('Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f'  %(epoch, train_running_loss /num_batches, train_acc/num_batches))
    #hist[epoch] = loss_item


print("Training loss: " + hist)