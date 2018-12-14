## Music Genre Classification with LSTMs

 * Classify music files based on genre from the GTZAN music corpus
 * GTZAN corpus is included for easy of use
 * Use multiple layers of LSTM Recurrent Neural Nets
 * Implementations in PyTorch, Keras & Darknet.

### Test trained LSTM model
 In ./weights/ you can find trained model weights and model architecture.
 To test model on your custom file, run
    python3 predict_example path/to/custom/file.mp3
 or to test model on our custom files, run
    python3 predict_example audios/classical_music.mp3

### Audio features extracted
 * [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
 * [Spectral Centroid](https://en.wikipedia.org/wiki/Spectral_centroid)
 * [Chroma](http://labrosa.ee.columbia.edu/matlab/chroma-ansyn/)
 * [Spectral Contrast](http://ieeexplore.ieee.org/document/1035731/)

### Dependencies
 * [Keras](https://keras.io) or [PyTorch](http://pytorch.org)
 * numpy
 * librosa - for audio feature extraction

### Ideas for improving accuracy:
 * Normalize MFCCs & other input features ([Recurrent BatchNorm](https://arxiv.org/pdf/1603.09025v4.pdf)?)
 * Decay learning rate
 * How are we initing the weights?
 * Better optimization hyperparameters (too little dropout)
 * Do you have avoidable bias? How's your variance?

### Accuracy

 * Training (at Epoch 400):
    Training loss: 0.5801
    Training accuracy: 0.7810

 * Validating:
    Dev loss:   0.734523485104
    Dev accuracy:   0.766666688025

 * Testing:
    Test loss:   0.900845060746
    Test accuracy:   0.683333342274