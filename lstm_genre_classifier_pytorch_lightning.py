#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    PyTorch lightning implementation of a simple 2-layer-deep LSTM for genre classification of musical audio.
    Feeding the LSTM stack are spectral {centroid, contrast}, chromagram & MFCC features (33 total values)

    We use this toy dataset and a well-understood, "easy" task to learn PTL based on the original PyTorch impl

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)




# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        if torch.cuda.is_available():
            print("\nTraining on GPU")
        else:
            print("\nNo GPU, training on CPU")

    def forward(self, input, hidden=None):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input, hidden)
        logits = self.linear(lstm_out[-1])
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, hidden


class MusicGenreClassifer(pl.LightningModule):

    def __init__(self, batch_size):
        super().__init__()
        self.model = LSTM(input_dim=33, hidden_dim=128, output_dim=8, num_layers=2)
        self.hidden = None

        # if you want to keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
        self.stateful = False

    def forward(self, x, hidden=None):
        prediction, self.hidden = self.model(x, hidden)
        return prediction

    def training_step(self, batch, batch_idx):
        # train_X shape: (total # of training examples, sequence_length, input_dim)
        # train_Y shape: (total # of training examples, # output classes)
        #
        # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
        # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)

        # IOHAVOC here, how to format data for easy use in datamodule?
        X_local_minibatch, y_local_minibatch = (
            train_X[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, ],
            train_Y[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, ],
        )

        # Reshape input & targets to "match" what the loss_function wants
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

        # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

        y_pred, self.hidden = self.model(X_local_minibatch, self.hidden)  # forward pass

        if not self.do_continue_train:
            self.hidden = None
        else:
            h_0, c_0 = self.hidden
            h_0.detach_(), c_0.detach_()
            self.hidden = (h_0, c_0)

        loss = nn.NLLLoss(y_pred, y_local_minibatch)  # compute loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.hidden = None

        # IOHAVOC here, how to format data for easy use in datamodule?
        for i in range(num_dev_batches):
            X_local_validation_minibatch, y_local_validation_minibatch = (
                dev_X[i * self.batch_size: (i + 1) * self.batch_size, ],
                dev_Y[i * self.batch_size: (i + 1) * self.batch_size, ],
            )
            X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
            y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

            y_pred, self.hidden = self.model(X_local_minibatch, self.hidden)
            if not self.stateful:
                hidden = None

            val_loss = nn.NLLLoss(y_pred, y_local_minibatch)
            return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer


class MusicGenreDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=35):
        super().__init__()
        self.batch_size = batch_size
        self.genre_features = GenreFeatureData()

    def prepare_data(self):
        # if all of the preprocessed files do not exist, regenerate them all for self-consistency
        if (
                os.path.isfile(self.genre_features.train_X_preprocessed_data)
                and os.path.isfile(self.genre_features.train_Y_preprocessed_data)
                and os.path.isfile(self.genre_features.dev_X_preprocessed_data)
                and os.path.isfile(self.genre_features.dev_Y_preprocessed_data)
                and os.path.isfile(self.genre_features.test_X_preprocessed_data)
                and os.path.isfile(self.genre_features.test_Y_preprocessed_data)
        ):
            print("Preprocessed files exist, deserializing npy files")
            self.genre_features.load_deserialize_data()
        else:
            print("Preprocessing raw audio files")
            self.genre_features.load_preprocess_data()

    def setup(self, stage: Optional[str] = None):

        train_X = torch.from_numpy(self.genre_features.train_X).type(torch.Tensor)
        dev_X = torch.from_numpy(self.genre_features.dev_X).type(torch.Tensor)
        test_X = torch.from_numpy(self.genre_features.test_X).type(torch.Tensor)

        # Targets is a long tensor of size (N,) which tells the true class of the sample.
        train_Y = torch.from_numpy(self.genre_features.train_Y).type(torch.LongTensor)
        dev_Y = torch.from_numpy(self.genre_features.dev_Y).type(torch.LongTensor)
        test_Y = torch.from_numpy(self.genre_features.test_Y).type(torch.LongTensor)

        # Convert {training, test} torch.Tensors
        print("Training X shape: " + str(self.genre_features.train_X.shape))
        print("Training Y shape: " + str(self.genre_features.train_Y.shape))
        print("Validation X shape: " + str(self.genre_features.dev_X.shape))
        print("Validation Y shape: " + str(self.genre_features.dev_Y.shape))
        print("Test X shape: " + str(self.genre_features.test_X.shape))
        print("Test Y shape: " + str(self.genre_features.test_Y.shape))

        # IOHAVOC
        self.train, self.val, self.test = load_datasets()
        self.train_dims = self.train.next_batch.size()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

# batch_size = 35  # num of training examples per minibatch
# num_epochs = 400

# if you want to keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
# stateful = False

# all training data (epoch) / batch_size == num_batches (12)
# num_batches = int(train_X.shape[0] / batch_size)
# num_dev_batches = int(dev_X.shape[0] / batch_size)


if __name__ == "__main__":
    model = MusicGenreClassifer()
    trainer = pl.Trainer()

    genre_dm = MusicGenreDataModule()

    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, genre_dm)