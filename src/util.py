import tensorflow as tf
import os
import random
import numpy as np


def select_audio(audio, sample_rate, s):
    samples = sample_rate * s
    start = random.choice(range(len(audio)-samples))

    return audio[start:start+samples]


class CustomDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, data_path,
                 data_files, dim, 
                 max_k, batch_size,
                 shuffle=True):
        
        self.data_path = data_path
        self.data_files = data_files
        self.dim = dim
        self.max_k = max_k
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.data_files)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_files)
    
    def __get_data(self, batches):
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, self.max_k),  dtype=int)

        for i, batch in enumerate(batches):
            X[i,] = self.__getX(batch)
            y[i,] = self.__getY(batch)

        return X, y

    def __getX(self, file_name):
        path = os.path.join(self.data_path, file_name)
        audio_file = tf.io.read_file(path)
        audio, sample_rate = tf.audio.decode_wav(contents=audio_file)
        audio = tf.squeeze(audio, axis=-1)
        audio = select_audio(audio, sample_rate, 5)

        waveform = tf.cast(audio, dtype=tf.float32)
        samples = int(sample_rate.numpy() * (25/1000))
        overlap = int(sample_rate.numpy() * (10/1000))
        spectrogram = tf.signal.stft(waveform, frame_length=samples, frame_step=overlap, fft_length=samples, pad_end=True)
        spectrogram = tf.abs(spectrogram) # not sure
        spectrogram = spectrogram[..., tf.newaxis]

        return spectrogram

    def __getY(self, file_name):
        k_speakers = int(file_name[:2].lstrip('0'))
        k_speakers_one_hot = tf.one_hot(k_speakers, self.max_k)

        return k_speakers_one_hot

    def __getitem__(self, index):
        batches = self.data_files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        

        return X, y
    
    def __len__(self):
        return self.n // self.batch_size