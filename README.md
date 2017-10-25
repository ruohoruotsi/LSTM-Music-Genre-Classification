## Music Genre Classification with LSTMs

 * Classify music files based on genre from the GTZAN music corpus
 * GTZAN corpus is included for easy of use
 * Use multiple layers of LSTM Recurrent Neural Nets
 * Implementations in PyTorch, Keras & Darknet.

#### Dependencies
 * Keras
 * numpy - yay math!
 * librosa - for audio feature extraction

Ideas for improving accuracy:
 * Normalize mfccs and other input features
 * Decay learning rate
 * How are we initing the weights?
 * Better optimization hyperparameters (too little dropout)
 * Do you have avoidable bias? How's your variance?