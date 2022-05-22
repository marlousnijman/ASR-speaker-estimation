from util import *
import argparse
import json
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

def get_train_mean_std(input_dir):
    """
    Compute the mean and standard deviation over
    the training set to be used for standardization
    over both the training set, as well as
    validation and test sets. 
    """
    spectrograms = []

    for filepath, filename in tqdm(get_files(input_dir, file_format=".wav")):
        # Read resampled audio
        audio_file = tf.io.read_file(filepath)
        audio, sample_rate = tf.audio.decode_wav(contents=audio_file)
        audio = tf.squeeze(audio, axis=-1)
        waveform = tf.cast(audio, dtype=tf.float32)

        # Convert to frequency domain
        samples = int(sample_rate.numpy() * (25/1000))
        overlap = int(sample_rate.numpy() * (10/1000))
        spectrogram = tf.signal.stft(waveform, frame_length=samples, frame_step=overlap, fft_length=samples, pad_end=True)
        spectrogram = tf.abs(spectrogram) 

        # Normalize 
        spectrogram, _ = tf.linalg.normalize(spectrogram, 'euclidean')

        # Collect all spectrograms
        spectrograms.append(spectrogram.numpy())

    # Return mean and standard deviation
    return np.mean(spectrograms), np.std(spectrograms)


def main(args):
    """
    Compute scaling parameters (mean and standard deviation)
    from the training set.
    """
    input_dir = os.path.join(args.data_dir, args.input_dir)
    mean, std = get_train_mean_std(input_dir)

    scaling_parameters = {"mean": float(mean),
                          "std": float(std)}

    output_file = os.path.join(args.data_dir, args.output_file)

    with open(output_file, 'w') as f:
        json.dump(scaling_parameters, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute scaling parameters from the training set')

    # Paths
    parser.add_argument('-d', '--data_dir', type=str, help="Data directory", default = "../../data/")
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default = "dataset/train")
    parser.add_argument('-o', '--output_file', type=str, help="output directory", default = "scaling_parameters.json")

    args = parser.parse_args()

    main(args)

