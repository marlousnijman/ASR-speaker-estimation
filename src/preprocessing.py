import numpy as np
import os
import soundfile as sf
import webrtcvad
import argparse
import util


def get_frames(audio, sample_rate, ms=10):
    """
    Get all frames of the audio sample with a certain sample rate
    of a certain duration in ms.
    """
    n = int(sample_rate * (ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0

    frames = []
    while offset + n < len(audio):
        frames.append(audio[offset:offset + n])
        timestamp += duration
        offset += n

    return frames


def get_start_frames(frame_ids, start_id=0):
    """
    Get the first consecutive frame IDs without speech.
    """
    start_frame_ids = []
    start = start_id
    for frame_id in frame_ids:
        if frame_id == start:
            start_frame_ids.append(frame_id)
            start += 1
    
    return start_frame_ids


def get_end_frames(frame_ids, end_id):
    """
    Get the final consecutive frame IDs without speech.
    """
    end_frame_ids = []
    end = end_id
    
    frame_ids.reverse()
    for frame_id in frame_ids:
        if frame_id == end:
            end_frame_ids.append(frame_id)
            end -= 1
    
    return end_frame_ids


def remove_speechless_frames(frames, sample_rate):
    """
    Remove part of the audio at the start and end 
    without speech.
    """
    # Initialize voice activity detector
    vad = webrtcvad.Vad(0)

    # Get ids of speechless frames
    no_voice_frame_ids = []
    for i, frame in enumerate(frames):
        if(vad.is_speech(frame, sample_rate) == False):
            no_voice_frame_ids.append(i)

    # Only remove frames at start and end of sample
    if len(no_voice_frame_ids) > 0:
        start = max(get_start_frames(no_voice_frame_ids), default=0)
        end = min(get_end_frames(no_voice_frame_ids, len(frames)-1), default=len(frames))
        frames = frames[start:end]

    # Flatten list
    new_audio = np.asarray([item for sublist in frames for item in sublist])

    # Return processed audio
    return new_audio


def get_chunks(audio, sample_rate, s):
    """
    Get chunks of audio with a length of s.
    """
    n = s * sample_rate
    offset = 0
    
    chunks = []
    while offset + n < len(audio):
        chunks.append(audio[offset:offset + n])
        offset += n

    return chunks


def preprocess(input_dir, output_dir, s):
    """
    Preprocess data by removing start and end silence 
    and convert flac to wav file.
    """
    for filepath, filename in util.get_files(input_dir):
        # Read audio
        audio, sample_rate = sf.read(filepath)

        # Get frames of 10 ms from the audio samples to remove 
        # speechless parts at beginning and end of audio
        frames = get_frames(audio, sample_rate)
        new_audio = remove_speechless_frames(frames, sample_rate)

        # Check if duration of sample is longer than 10s
        # to make chunks.
        duration = len(new_audio) / sample_rate
        if duration >= s:
            chunks = get_chunks(new_audio, sample_rate, s)

            # Write processed audio chunks to file
            for i, chunk in enumerate(chunks):
                new_filename = filename.replace(".flac", f"-{i:04}.wav")
                sf.write(os.path.join(output_dir, new_filename), chunk, sample_rate)


def main(args):
    """
    Read .flac audio files, remove start and end silence, and
    create chunks of a certain duration and write as .wav files
    """
    input_dir = os.path.join(args.data_dir, args.input_dir)
    output_dir = os.path.join(args.data_dir, args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess(input_dir, output_dir, args.seconds)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess LibriSpeech data for CountNet')

    # Paths
    parser.add_argument('-d', '--data_dir', type=str, help="Data directory", default = "../../data/")
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default = "dev-clean/LibriSpeech/dev-clean/")
    parser.add_argument('-o', '--output_dir', type=str, help="output directory", default = "processed/")

    # Parameters
    parser.add_argument('-s', '--seconds', type=int, help="Length of chunks in seconds", default = 10)

    args = parser.parse_args()

    main(args)