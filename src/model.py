import tensorflow as tf
from tensorflow.keras import layers, models


def CRNN(input_shape, max_k):
    model = models.Sequential()

    # CNN
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(0.25))

    # LSTM
    model.add(layers.Reshape((53, -1)))
    model.add(layers.LSTM(40, return_sequences=True))
    model.add(layers.MaxPooling1D((2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(max_k)) #+1
    model.add(layers.Softmax())

    return model