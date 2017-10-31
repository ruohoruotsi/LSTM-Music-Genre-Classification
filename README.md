## Music Genre Classification with LSTMs

 * Classify music files based on genre from the GTZAN music corpus
 * GTZAN corpus is included for easy of use
 * Use multiple layers of LSTM Recurrent Neural Nets
 * Implementations in PyTorch, Keras & Darknet.

#### Dependencies
 * Keras
 * numpy - yay math!
 * librosa - for audio feature extraction

#### Ideas for improving accuracy:
 * Normalize mfccs and other input features
 * Decay learning rate
 * How are we initing the weights?
 * Better optimization hyperparameters (too little dropout)
 * Do you have avoidable bias? How's your variance?

#### Accuracy

 * Training (at Epoch 400):
    Training loss: 0.5801
    Training accuracy: 0.7810

 * Validating:
    Dev loss:   0.734523485104
    Dev accuracy:   0.766666688025

 * Testing:
    Test loss:   0.900845060746
    Test accuracy:   0.683333342274