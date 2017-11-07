import os
import re
import librosa
import numpy as np
import math

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam

# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

class GenreFeatureData:

    'LSTM-based music genre classification'
    hop_length = None
    genre_list = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

    dir_trainfolder = "./gtzan/_train"
    dir_devfolder = "./gtzan/_validation"
    dir_testfolder = "./gtzan/_test"
    dir_all_files = "./gtzan"

    train_X_preprocessed_data = 'data_train_input.npy'
    train_Y_preprocessed_data = 'data_train_target.npy'
    dev_X_preprocessed_data = 'data_validation_input.npy'
    dev_Y_preprocessed_data = 'data_validation_target.npy'
    test_X_preprocessed_data = 'data_test_input.npy'
    test_Y_preprocessed_data = 'data_test_target.npy'

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []

    def load_preprocess_data(self):
        self.trainfiles_list = self.path_to_audiofiles(self.dir_trainfolder)
        self.devfiles_list = self.path_to_audiofiles(self.dir_devfolder)
        self.testfiles_list = self.path_to_audiofiles(self.dir_testfolder)

        all_files_list = []
        all_files_list.extend(self.trainfiles_list)
        all_files_list.extend(self.devfiles_list)
        all_files_list.extend(self.testfiles_list)

        #self.precompute_min_timeseries_len(all_files_list)
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set
        self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list)
        with open(self.train_X_preprocessed_data, 'wb') as f:
            np.save(f, self.train_X)
        with open(self.train_Y_preprocessed_data, 'wb') as f:
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)

        # Validation set
        self.dev_X, self.dev_Y = self.extract_audio_features(self.devfiles_list)
        with open(self.dev_X_preprocessed_data, 'wb') as f:
            np.save(f, self.dev_X)
        with open(self.dev_Y_preprocessed_data, 'wb') as f:
            self.dev_Y = self.one_hot(self.dev_Y)
            np.save(f, self.dev_Y)

        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
        with open(self.test_X_preprocessed_data, 'wb') as f:
            np.save(f, self.test_X)
        with open(self.test_Y_preprocessed_data, 'wb') as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)

    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.dev_X = np.load(self.dev_X_preprocessed_data)
        self.dev_Y = np.load(self.dev_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def precompute_min_timeseries_len(self, list_of_audiofiles):
        for file in list_of_audiofiles:
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_length))

    def extract_audio_features(self, list_of_audiofiles):
        #timeseries_length = min(self.timeseries_length_list)
        timeseries_length = 128
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)

            splits = re.split('[ .]', file)
            genre = re.split('[ /]', splits[1])[3]
            target.append(genre)

            # data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            # #data[i, :, 0:1] = spectral_center.T[0:timeseries_length, :]
            # #data[i, :, 1:13] = chroma.T[0:timeseries_length, :]
            # #data[i, :, 13:20] = spectral_contrast.T[0:timeseries_length, :]

            # save
            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

            print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

        return data, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio


genre_features = GenreFeatureData()
#genre_features.load_preprocess_data()
genre_features.load_deserialize_data()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 35
nb_epochs = 400

print("Training X shape: " + str(genre_features.train_X.shape))
print("Training Y shape: " + str(genre_features.train_Y.shape))
print("Dev X shape: " + str(genre_features.dev_X.shape))
print("Dev Y shape: " + str(genre_features.dev_Y.shape))
print("Test X shape: " + str(genre_features.test_X.shape))
print("Test Y shape: " + str(genre_features.test_X.shape))

input_shape = (genre_features.train_X.shape[1], genre_features.train_X.shape[2])
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=genre_features.train_Y.shape[1], activation='softmax'))

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(genre_features.train_X, genre_features.train_Y, batch_size=batch_size, epochs=nb_epochs)

print("\nValidating ...")
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


print("\nTesting ...")
score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)