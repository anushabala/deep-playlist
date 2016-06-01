'''
Some of the code here (particularly to convert audio to numpy arrays) is borrowed from Matt Vitelli's
excellent GRUV repository at https://github.com/MattVitelli/GRUV.
'''

__author__ = 'anushabala'
from pydub import AudioSegment
import os
import scipy.io.wavfile as wav
import numpy as np
from config import DataConfig, ModelConfig
from pydub.exceptions import CouldntDecodeError
from collections import defaultdict
import pickle
import warnings

def convert_mp3_to_wav(mp3_path, wav_path, max_length=10):
    try:
        if os.path.exists(wav_path):
            return True #todo ideally check here whether file is valid but blah for now
        sound = AudioSegment.from_mp3(mp3_path)
        sound = sound[:max_length*1000]
        sound.export(wav_path, format='wav')
    except (IOError, CouldntDecodeError) as ie:
        print "Error while converting mp3 to wav: %s" % mp3_path
        return False

    return True


def convert_dataset_to_wav(mp3_dir, wav_dir):
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    for mp3_file in os.listdir(mp3_dir):
        name = mp3_file.replace(".mp3", "")
        convert_mp3_to_wav(os.path.join(mp3_dir, name), os.path.join(wav_dir, name))


def read_wav_as_nparray(filename):
    data = wav.read(filename)
    sample_rate = data[0]
    audio = data[1]
    audio = audio.astype('float32') / 32767.0
    # take mean across channels? todo
    if len(audio.shape) == 2 and audio.shape[1] != 1:
        audio = np.mean(audio, axis=1)
    return audio, sample_rate


def convert_np_audio_to_blocks(data, block_size, max_blocks):
    block_lists = []
    total_samples = data.shape[0]
    num_samples_so_far = 0
    while num_samples_so_far < total_samples and len(block_lists) < max_blocks:
        block = data[num_samples_so_far: num_samples_so_far + 2]
        if block.shape[0] < block_size:
            padding = np.zeros((block_size - block.shape[0],))
            block = np.concatenate((block, padding))
        block_lists.append(block)
        num_samples_so_far += block_size

    while len(block_lists) < max_blocks:
        block_lists.append(-1 * np.ones((block_size,)))

    return block_lists


def time_blocks_to_fft_blocks(blocks):
    fft_blocks = []
    for block in blocks:
        fft_block = np.fft.fft(block)
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(new_block)

    return fft_blocks


def load_song_as_np(filename, block_size=11025, fft=True, max_blocks=15):
    data, sample_rate = read_wav_as_nparray(filename)
    blocks = convert_np_audio_to_blocks(data, block_size, max_blocks=max_blocks)
    if not fft:
        return blocks
    fft_blocks = time_blocks_to_fft_blocks(blocks)
    return fft_blocks


def get_audio_data(audio_config, directory, wav_dir, filter_set=None, max_files=-1):
    ctr = 0
    max_seq_len = audio_config.max_seq_len
    all_data = {}
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    for name in os.listdir(directory):
        if ".mp3" not in name:
            continue

        track_id = name.replace(".mp3", "")
        if filter_set is not None and track_id not in filter_set:
            continue
        mp3_path = os.path.join(directory, name)
        wav_path= os.path.join(wav_dir, track_id) + '.wav'
        conversion_success = convert_mp3_to_wav(mp3_path, wav_path)
        if not conversion_success:
            continue

        song_blocks = load_song_as_np(wav_path, block_size=audio_config.block_size, fft=audio_config.use_fft, max_blocks=max_seq_len)
        all_data[track_id] = song_blocks

        ctr += 1
        if ctr % 50 == 0:
            print "Finished getting data for %d tracks" % ctr
        if ctr >= max_files > 0:
            break

    return all_data


def unpickle_data(pickle_file):
    (X1, X2, labels) = pickle.load(open(pickle_file, 'rb'))
    return X1, X2, labels


def get_tracks_present(mappings_file, max_examples_per_track=2):
    f = open(mappings_file, 'r')
    examples_per_track = defaultdict(int)
    all_tracks = set()
    for line in f.readlines():
        line = line.strip().split(',')
        first_id = line[0]
        second_id = line[1]
        if examples_per_track[first_id] >= max_examples_per_track:
            continue

        examples_per_track[first_id] += 1
        all_tracks.add(first_id)
        all_tracks.add(second_id)

    f.close()
    return all_tracks


def get_dataset(data_config, audio_config, max_examples_per_track=2):
    # todo reload and update data
    save_data = data_config.save_data
    if not data_config.reload:
        if os.path.exists(data_config.pickle_file):
            if audio_config.verbose:
                X1, X2, labels = unpickle_data(data_config.pickle_file)
                print "Loaded dataset from data stored at %s. Data shape: %s" % (data_config.pickle_file, str(X1.shape))
                return X1, X2, labels
            return unpickle_data(data_config.pickle_file)

        else:
            warnings.warn("Warning: Pickled data file (%s) does not exist. Reloading data instead." % data_config.pickle_file)

    if audio_config.verbose:
        print "Reloading data from mp3 dir: %s" % data_config.mp3_dir
    mappings_file_path = data_config.mappings_file
    if audio_config.verbose:
        print "Loading list of tracks present in mappings.."
    all_tracks = get_tracks_present(mappings_file_path, max_examples_per_track)
    if audio_config.verbose:
        print "Loading track --> np array mappings..."
    data = get_audio_data(audio_config, data_config.mp3_dir, data_config.wav_dir, filter_set=all_tracks, max_files=data_config.max_files)
    if audio_config.verbose:
        print "Finished loading track --> np array mappings for %d tracks." % len(data.keys())

    # mappings_file_path, data, max_examples_per_track=2, save_data=True, save_path='../data/audio/train_data.pkl'

    max_examples = data_config.max_examples
    X1 = []
    X2 = []
    labels = []
    mappings_file = open(mappings_file_path, 'r')
    examples_per_track = defaultdict(int)
    if audio_config.verbose:
        print "Loading dataset with labels.."
    for line in mappings_file.readlines():
        line = line.strip().split(',')
        first_id = line[0]
        second_id = line[1]
        sim_score = float(line[2])

        if first_id in data.keys() and second_id in data.keys():
            # if examples_per_track[first_id] == 0:
            #     if sim_score < 0.5: # if there is no similar track for a track, don't add it to the training data
            #         examples_per_track[first_id] = -1

            # skip tracks with no similar tracks
            if examples_per_track[first_id] == -1:
                continue

            if examples_per_track[first_id] <= max_examples_per_track:
                X1.append(data[first_id])
                X2.append(data[second_id])
                sim_score = 1 if sim_score >= 0.5 else 0
                labels.append(sim_score)
                examples_per_track[first_id] += 1

                if len(X1) % 50 == 0 and audio_config.verbose:
                    print "Dataset size: %d" % len(X1)

        if 0 < max_examples <= len(X1):
            break

    # todo normalize data
    X1 = np.array(X1)
    X2 = np.array(X2)
    labels = np.array(labels)

    if save_data:
        if audio_config.verbose:
            print "Saving loaded data to %s" % data_config.pickle_file
        f_out = open(data_config.pickle_file, 'wb')
        pickle.dump((X1, X2, labels), f_out)
        f_out.close()

    return X1, X2, labels


def test():
    config = ModelConfig()
    data = get_audio_data(config, '../data/audio/train', '../data/audio/train_wav')
    data_config = DataConfig()
    X1, X2, labels = get_dataset(data_config, config)

if __name__ == "__main__":
    test()
