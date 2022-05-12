import tensorflow as tf
import os
import random
import numpy as np
import librosa
import soundfile as sf


def get_files(data_dir, file_format=".flac"):
    """
    Get the file names of all files of a certain format that are available
    in the data directory
    """
    filepaths = []
    files = []

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(file_format):

                # Save both the path and the file name
                filepaths.append(os.path.join(dirpath, filename))
                files.append(filename)

    return list(zip(filepaths, files))


def select_audio(audio, sample_rate, s):
    """
    Randomly select audio sample with a duration of s seconds
    from a longer audio segment.
    """
    samples = sample_rate * s
    start = random.choice(range(len(audio)-samples))

    return audio[start:start+samples]


class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator used for loading in audio segments
    and transforming them to the time-frequency domain.
    """
    
    def __init__(self, data_path,
                 data_files, dim, 
                 max_k, batch_size,
                 shuffle=True):
        
        self.data_path = data_path
        self.data_files = data_files
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.data_files)
        self.max_k = max_k + 1
    
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
        k_speakers_str = file_name[:2].lstrip('0')
        k_speakers = 0 if len(k_speakers_str) == 0 else int(k_speakers_str)
        k_speakers_one_hot = tf.one_hot(k_speakers, self.max_k)

        return k_speakers_one_hot

    def __getitem__(self, index):
        batches = self.data_files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        

        return X, y
    
    def __len__(self):
        return self.n // self.batch_size


def process_noise(data_dir, dataset, noise_samples, sample_rate):
    """
    Process noise samples by resampling the audio to the same
    sample rate as the speaker audio and saving them in the
    correct dataset directories.
    """
    for i, noise_sample in enumerate(noise_samples):
        # Load audio with new sample rate
        audio, samplerate = librosa.load(noise_sample[0], sr=sample_rate)

        # Save new audio 
        new_name = os.path.join(data_dir, f"{dataset}/", f"{0:02}_speakers_{i}.wav")
        sf.write(new_name, audio, sample_rate)
