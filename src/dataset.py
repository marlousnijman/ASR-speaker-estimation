import argparse
import os
import random
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def make_dataset(input_dir, output_dir, max_k):
    """
    Make a dataset by combining audio files for different
    numbers of speakers.
    """
    all_speakers = os.listdir(input_dir)

    # Compute how many samples per k we can make
    # with the available dataset
    n = np.sum(np.arange(max_k+1))
    samples = len(all_speakers) // n

    for k in range(1, max_k+1):
        for s in range(samples):
            speakers = random.sample(all_speakers, k)
            all_audio = []

            # Collect k random speakers
            for speaker in speakers:
                audio, sample_rate = sf.read(os.path.join(input_dir, speaker))
                all_audio.append(audio)

                # Remove used samples
                all_speakers.remove(speaker)

            # Combine audio
            all_audio = np.sum(all_audio, axis=0)

            # Peak normalize audio to 0 dB
            normalized_audio = pyln.normalize.peak(all_audio, 0)

            # Write to file
            sf.write(os.path.join(output_dir, f"{k:02}_speakers_{s}.wav"), normalized_audio, sample_rate)

    return samples


def main(args):
    """
    Create a dataset with an equal number of samples for
    different number of speakers and normalize the volume.
    """
    input_dir = os.path.join(args.data_dir, args.input_dir)
    output_dir = os.path.join(args.data_dir, args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    make_dataset(input_dir, output_dir, args.max_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess LibriSpeech data for CountNet')

    # Paths
    parser.add_argument('-d', '--data_dir', type=str, help="Data directory", default = "../../data/")
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default = "processed/")
    parser.add_argument('-o', '--output_dir', type=str, help="output directory", default = "dataset/")

    # Parameters
    parser.add_argument('-k', '--max_k', type=int, help="Maximum number of speakers", default = 10)

    args = parser.parse_args()

    main(args)
