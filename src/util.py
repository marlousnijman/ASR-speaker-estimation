import tensorflow as tf
import os
import random
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import pandas as pd
from tqdm import tqdm


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


def get_speakers(input_dir, file_format=".flac"):
    """
    Get a list of all unique speakers.
    """
    speakers = []
    for filepath, filename in get_files(input_dir, file_format):
        speaker_id = filename[:filename.index("-")]
        speakers.append(speaker_id)

    return list(np.unique(speakers))


def select_audio(audio, sample_rate, s, train=True):
    """
    Randomly select audio sample with a duration of s seconds
    from a longer audio segment.
    """
    samples = sample_rate * s
    start = random.choice(range(len(audio)-samples))

    if train:
        audio_sample = audio[start:start+samples]
    else:
        audio_sample = audio[:samples]

    return audio_sample


class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator used for loading in audio segments
    and transforming them to the time-frequency domain.
    """
    
    def __init__(self, data_path,
                 data_files, dim, 
                 max_k, batch_size,
                 mean, std, s,
                 train=True, 
                 shuffle=True):
        
        self.data_path = data_path
        self.data_files = data_files
        self.dim = dim
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.s = s
        self.shuffle = shuffle
        self.train = train
        
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
        # Read audio
        path = os.path.join(self.data_path, file_name)
        audio_file = tf.io.read_file(path)
        audio, sample_rate = tf.audio.decode_wav(contents=audio_file)
        audio = tf.squeeze(audio, axis=-1)
        
        # Select (random) s seconds sample
        audio = select_audio(audio, sample_rate, self.s, train=self.train)

        # Convert to frequency domain
        waveform = tf.cast(audio, dtype=tf.float32)
        samples = int(sample_rate.numpy() * (25/1000))
        overlap = int(sample_rate.numpy() * (10/1000))
        spectrogram = tf.signal.stft(waveform, frame_length=samples, frame_step=overlap, fft_length=samples, pad_end=True)
        spectrogram = tf.abs(spectrogram) 

        # Normalize
        spectrogram, _ = tf.linalg.normalize(spectrogram, 'euclidean')

        # Standardize
        spectrogram = (spectrogram - self.mean) / self.std

        # Add batch dimension
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


def process_noise(data_dir, dataset, noise_samples, sample_rate, duration=10):
    """
    Process noise samples by resampling the audio to the same
    sample rate as the speaker audio and saving them in the
    correct dataset directories.
    """
    for i, noise_sample in tqdm(enumerate(noise_samples)):
        # Load audio with new sample rate
        audio, samplerate = librosa.load(noise_sample[0], sr=sample_rate, duration=duration)

        # Peak normalize
        normalized_audio = pyln.normalize.peak(audio, 0)

        # Save new audio 
        new_name = os.path.join(data_dir, f"{dataset}/", f"{0:02}_speakers_{i}.wav")
        sf.write(new_name, normalized_audio, sample_rate)


def remove_speech_noise(files, meta_path, speech_categories):
    """
    Remove all noise files that contain speech based
    on the speech categories mentioned in the paper. 
    """
    # Read meta data
    noise_df = pd.read_csv(meta_path, sep="\t", header=None, names=["filename", "category", "code"])

    # Remove path
    noise_df["filename"] = noise_df["filename"].str.replace("audio/", "")

    # Create regular expression
    speech_categories = "|".join(speech_categories)                    

    # Keep rows that contain the speech categories
    noise_df = noise_df[noise_df["category"].str.contains(speech_categories)]

    # Get speech-containing noise filenames
    speech_files = noise_df["filename"].tolist()

    # Remove from files
    for f in files:
        if f[1] in speech_files:
            files.remove(f)

    return files