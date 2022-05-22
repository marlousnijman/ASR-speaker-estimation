import argparse
import os
import random
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import util
from tqdm import tqdm


def make_dataset(input_dir, output_dir, max_k, samples=1820, train=False):
    """
    Create a dataset with an equal number of samples for each
    k = 0, ..., max_k, by adding k speech samples from randomly
    selected unique speakers, and peak normalize the resulting
    audio.
    """
    speakers = util.get_speakers(input_dir, file_format=".wav")
    files = os.listdir(input_dir)

    for k in tqdm(range(1, max_k+1)):
        for s in range(samples):
            # Select k random speakers
            k_speakers = random.sample(speakers, k)
            all_audio = []

            # Select speaker audio
            for speaker in k_speakers:
                speaker_files = [f for f in files if f.startswith(speaker)]

                # Select random audio file
                speaker_file = random.choice(speaker_files)

                # Remove used data during training
                if train: 
                    # Remove speakers with no remaining audio samples
                    if len(speaker_files) == 1:
                        speakers.remove(speaker)
                
                    # Remove audio file from list
                    files.remove(speaker_file)

                # Read audio
                audio, sample_rate = sf.read(os.path.join(input_dir, speaker_file))
                all_audio.append(audio)

            # Combine audio
            all_audio = np.sum(all_audio, axis=0)

            # Peak normalize audio to 0 dB
            normalized_audio = pyln.normalize.peak(all_audio, 0)

            # Write to file
            sf.write(os.path.join(output_dir, f"{k:02}_speakers_{s}.wav"), normalized_audio, sample_rate)


def main(args):
    """
    Create a dataset with an equal number of samples for
    different number of speakers and normalize the volume.
    """
    input_dir = os.path.join(args.data_dir, args.input_dir)
    output_dir = os.path.join(args.data_dir, args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    make_dataset(input_dir, output_dir, args.max_k, args.samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess LibriSpeech data for CountNet')

    # Paths
    parser.add_argument('-d', '--data_dir', type=str, help="Data directory", default = "../../data/")
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default = "processed/")
    parser.add_argument('-o', '--output_dir', type=str, help="output directory", default = "dataset/")

    # Parameters
    parser.add_argument('-k', '--max_k', type=int, help="Maximum number of speakers", default = 10)
    parser.add_argument('-s', '--samples', type=int, help="Number of samples for each speaker", default = 1820)

    args = parser.parse_args()

    main(args)
