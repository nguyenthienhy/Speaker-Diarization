from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import random
import glob
import librosa as lr
import numpy as np
import time
import model, toolkits, utils

parser = argparse.ArgumentParser()
# Set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'pre_trained/weights.h5', type=str)
parser.add_argument('--data_path', default='../data/data_en_train', type=str)
parser.add_argument('--epochs', default=1000, type=int)

# Set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)

# Set up other params.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

args = parser.parse_args()


def load_wav(vid_path, sr):
    wav, _ = lr.load(vid_path, sr=sr)
    intervals = lr.effects.split(wav, top_db=20)  # VAD
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output)


def load_data(selected_wavs, win_length=400, sr=16000, hop_length=160, n_fft=512, min_win_time=240, max_win_time=1600):
    win_time = np.random.randint(min_win_time, max_win_time, 1)[0]  # win_length in [240, 1600] ms
    win_spec = win_time // (1000 // (sr // hop_length))  # win_length in spectrum
    hop_spec = win_spec // 2
    wavs = np.array([])
    change_points = []
    paths = list(zip(*selected_wavs))[0]
    speakers = list(zip(*selected_wavs))[1]
    for path in paths:
        wav = load_wav(path, sr=sr)
        wavs = np.concatenate((wavs, wav))
        change_points.append(wavs.shape[0] // hop_length)  # Change_point in spectrum
    linear_spectogram = utils.linear_spectogram_from_wav(wavs, hop_length, win_length, n_fft)
    mag, _ = lr.magphase(linear_spectogram)  # Magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    utterance_specs = []
    utterance_speakers = []
    cur_spec = 0
    cur_speaker = speakers[0]
    i = 0
    while True:
        if cur_spec + win_spec > time:
            break
        spec_mag = mag_T[:, cur_spec:cur_spec + win_spec]

        if cur_spec + win_spec // 2 > change_points[i]:  # cur win_spec span to the next speaker
            i += 1
            cur_speaker = speakers[i]
        # Preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterance_specs.append(spec_mag)
        utterance_speakers.append(cur_speaker)
        cur_spec += hop_spec
    return utterance_specs, utterance_speakers


def prepare_dataset(data_dir):
    paths_list = []
    speakers_labels_list = []
    for data_path in glob.glob(data_dir + "/*"):
        speakers_dir = glob.glob(data_path + "/*")
        for speaker_dir in speakers_dir:
            wav_path = glob.glob(speaker_dir + "\\*.wav")
            label = data_path.split("\\")[-1] + '_' + str(speaker_dir.split('\\')[-1])
            for wav in wav_path:
                utterance_path = wav
                paths_list.append(utterance_path)
                speakers_labels_list.append(label)
    return list(zip(paths_list, speakers_labels_list))


def generate_embeddings():
    # GPU config
    toolkits.initialize_GPU(args.gpu)

    # Construct the dataset generator
    vgg_params = {
        'dim': (257, None, 1),
        'nfft': 512,
        'min_slice': 720,
        'win_length': 400,
        'hop_length': 160,
        'n_classes': 5994,
        'sampling_rate': 16000,
        'normalize': True
    }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=vgg_params['dim'], num_class=vgg_params['n_classes'],
                                                mode='eval', params=args)

    if args.resume:
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            print(f'Successfully loading model {args.resume}')
        else:
            raise IOError(f'No checkpoint found at {args.resume}')
    else:
        raise IOError('Please type in the model to load')

    """The feature extraction process has to be done sample-by-sample,
       because each sample is of different lengths.
    """

    path_speaker_label_tuples = prepare_dataset(args.data_path)
    train_sequence = []
    train_cluster_id = []

    # Random choice utterances from whole wav files
    for epoch in range(args.epochs):
        # A merged utterance contains [10,20] utterances
        speakers_number = np.random.randint(10, 20, 1)[0]
        selected_speakers = random.sample(path_speaker_label_tuples, speakers_number)
        utterance_specs, utterance_speakers = load_data(selected_speakers, min_win_time=240, max_win_time=1600)

        feats = []
        for spec in utterance_specs:
            spec = np.expand_dims(np.expand_dims(spec, 0), -1)
            v = network_eval.predict(spec)
            feats += [v]

        feats = np.array(feats)[:, 0, :]  # [splits, embedding dim]

        train_sequence.append(feats)
        train_cluster_id.append(utterance_speakers)

        print(f'Epoch:{epoch}, utterance length: {len(utterance_speakers)}, speakers: {len(selected_speakers)}')

    if args.mode == 'train':
        npz_name = 'training_data'
    else:
        npz_name = 'testing_data'

    np.savez('../data/data_embedding/' + npz_name, train_sequence=train_sequence, train_cluster_id=train_cluster_id)


if __name__ == '__main__':
    start = time.time()
    generate_embeddings()
    end = time.time()
    print("Total time for embedding phase: " + str((end - start) / 60))
