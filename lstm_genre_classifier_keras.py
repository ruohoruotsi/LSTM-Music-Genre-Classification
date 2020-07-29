#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Keras implementation of a simple 2-layer-deep LSTM for genre classification of musical audio.
    Feeding the LSTM stack are spectral {centroid, contrast}, chromagram and MFCC features

    Model Summary:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    lstm_1 (LSTM)                (None, 128, 128)          82944
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 32)                20608
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 264
    =================================================================
    Total params: 103,816
    Trainable params: 103,816
    Non-trainable params: 0

    X shape (total # of training examples, sequence_length, input_dim)
    Y shape (total # of training examples, # output classes)
    ________________________________
    Training X shape: (420, 128, 33)
    Training Y shape: (420, 8)
    ________________________________
    Dev X shape: (120, 128, 33)
    Dev Y shape: (120, 8)
    ________________________________
    Test X shape: (60, 128, 33)
    Test Y shape: (60, 128, 33)

    420 is the total number of 30 second training files, across all genres. From each file, we
    extract a 128-length sequence of inputs (it could be longer). Each input is 33 element column
    vector comprising mfcc, centroid, contrast & chroma features.

    An epoch, containing all training data (420 sequences), is divided into 12 mini-batches of
    length 35, i.e. each mini-batch has 35 sequences. An LSTM RNN (stack) loops over each sequence
    for sequence_length (128) steps computing an output value (of dimension 32) which is
    transformed (via Dense layer) into 8 output classes
"""

import logging
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)

# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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

print("Training X shape: " + str(genre_features.train_X.shape))
print("Training Y shape: " + str(genre_features.train_Y.shape))
print("Dev X shape: " + str(genre_features.dev_X.shape))
print("Dev Y shape: " + str(genre_features.dev_Y.shape))
print("Test X shape: " + str(genre_features.test_X.shape))
print("Test Y shape: " + str(genre_features.test_Y.shape))

input_shape = (genre_features.train_X.shape[1], genre_features.train_X.shape[2])
print("Build LSTM RNN model ...")
model = Sequential()

model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=genre_features.train_Y.shape[1], activation="softmax"))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 35  # num of training examples per minibatch
num_epochs = 400
model.fit(
    genre_features.train_X,
    genre_features.train_Y,
    batch_size=batch_size,
    epochs=num_epochs,
)

print("\nValidating ...")
score, accuracy = model.evaluate(
    genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1
)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


print("\nTesting ...")
score, accuracy = model.evaluate(
    genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Creates a HDF5 file 'lstm_genre_classifier.h5'
model_filename = "lstm_genre_classifier_lstm.h5"
print("\nSaving model: " + model_filename)
model.save(model_filename)
# Creates a json file
print("creating .json file....")
model_json = model.to_json()
f = Path("./lstm_genre_classifier_lstm.json")
f.write_text(model_json)
