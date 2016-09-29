import os
import re
import librosa
import numpy as np
import math

# Keras deep learning library
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback


# class TestCallback(Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data
#
#     def on_epoch_end(self, epoch, logs={}):
#         x, y = self.test_data
#         loss, acc = self.model.evaluate(x, y, verbose=0)
#         print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#
# model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
#           callbacks=[TestCallback((X_test, Y_test))])


class GenreFeatureData:

    'LSTM-based music genre classification'
    hop_length = None
    genre_list = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

    dir_testfolder = "./gtzan/_test"
    dir_validationfolder = "./gtzan/_validation"
    dir_trainfolder = "./gtzan/_train"

    test_X = test_Y = None
    validation_X = validation_Y = None
    train_X = train_Y = None

    def __init__(self):
        self.hop_length = 512

    def load_aggregate_save_training_data(self):
        self.path_to_testfiles = path_to_audiofiles(self.dir_testfolder)
        self.path_to_validationfiles = path_to_audiofiles(self.dir_validationfolder)
        self.path_to_trainfiles = path_to_audiofiles(self.dir_trainfolder)

        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.path_to_testfiles, hop_length=self.hop_length)
        with open('data_test_input2.npy', 'wb') as f:
            np.save(f, self.test_X)
        with open('data_test_target2.npy', 'wb') as f:
            np.save(f, self.test_Y)

        # Validation set
        self.validation_X, self.validation_Y = self.extract_audio_features(self.path_to_validationfiles, hop_length=self.hop_length)
        with open('data_validation_input2.npy', 'wb') as f:
            np.save(f, self.validation_X)
        with open('data_validation_target2.npy', 'wb') as f:
            np.save(f, self.validation_Y)

        # Training set
        self.train_X, self.train_Y = self.extract_audio_features(self.path_to_trainfiles, hop_length=self.hop_length)
        with open('data_train_input2.npy', 'wb') as f:
            np.save(f, self.train_X)
        with open('data_train_target2.npy', 'wb') as f:
            np.save(f, self.train_Y)

    def load_saved_training_data(self):

        self.test_X = np.load('./data_test_input2.npy')
        self.test_Y = np.load('./data_validation_target2.npy')
        # self.test_target = self.one_hot(self.test_target)

        self.validation_X = np.load('./data_validation_input2.npy')
        self.validation_Y = np.load('./data_validation_target2.npy')
        self.validation_Y = self.one_hot(self.validation_Y)

        self.train_X = np.load('./data_train_input2.npy')
        self.train_Y = np.load('./data_train_target2.npy')
        self.train_Y = self.one_hot(self.train_Y)

    def extract_audio_features(self, list_of_audiofiles, hop_length=512):
        timeseries_length_list = []
        for file in list_of_audiofiles:
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            timeseries_length_list.append(math.ceil(len(y) / hop_length))

        timeseries_length = min(timeseries_length_list)
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 27), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            spectral_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)

            splits = re.split('[ .]', file)
            genre = re.split('[ /]', splits[1])[3]
            target.append(genre)

            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:27] = spectral_roll.T[0:timeseries_length, :]

            print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

        return data, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot


def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".au"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio


genre_features = GenreFeatureData()
# genre_features.load_aggregate_save_training_data()
genre_features.load_saved_training_data()

opt = Adam(lr=0.0067, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# opt = Adam(lr=0.01)
# opt = RMSprop()
# opt = SGD(nesterov=True)
# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

batch_size = 128
nb_epochs = 400

print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(input_dim=genre_features.validation_X.shape[2],
               output_dim=32, activation='sigmoid', dropout_U=0.05, dropout_W=0.05, return_sequences=True))
model.add(LSTM(output_dim=16, activation='sigmoid', dropout_U=0.05, dropout_W=0.05, return_sequences=False))
model.add(Dense(output_dim=genre_features.train_Y.shape[1], activation='softmax'))

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, nb_epoch=nb_epochs)

print("Evaluating ...")
score, accuracy = model.evaluate(genre_features.validation_X, genre_features.validation_Y,
                                 batch_size=batch_size, verbose=1)

print("Validation score: ", score)
print("Accuracy score:   ", accuracy)