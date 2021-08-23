"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

if __name__ == '__main__':
    
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
    import numpy as np
    import uisrnn
    import librosa
    import sys
    sys.path.append('embedding')
    sys.path.append('visualization')
    import toolkits
    import model as spkModel
    import consts
    import new_utils
    import glob
    import torch
    torch.cuda.empty_cache()

    # ===========================================
    #        Parse the argument
    # ===========================================

    import argparse
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='P100', type=str)
    parser.add_argument('--resume', default=r'embedding/pre_trained/weights.h5', type=str)
    parser.add_argument('--data_path', default='4persons', type=str)
    # set up network configuration.
    parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
    parser.add_argument('--ghost_cluster', default=2, type=int)
    parser.add_argument('--vlad_cluster', default=8, type=int)
    parser.add_argument('--bottleneck_dim', default=512, type=int)
    parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

    global args
    args = parser.parse_args()


    SAVED_MODEL_NAME = 'saved_model_uisrnn/model_en_0822.uisrnn'

    # gpu configuration
    toolkits.initialize_GPU(args.gpu)

    params = {'dim': (257, None, 1),
            'nfft': 512,
            'spec_len': 250,
            'win_length': 400,
            'hop_length': 160,
            'n_classes': 5994,
            'sampling_rate': 16000,
            'normalize': True,
            }
    
    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', params=args)
    network_eval.load_weights(args.resume, by_name=True)


    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)

    def load_wav(vid_path, sr):
        wav, _ = librosa.load(vid_path, sr=sr)
        intervals = librosa.effects.split(wav, top_db=20)
        wav_output = []
        for sliced in intervals:
            wav_output.extend(wav[sliced[0]:sliced[1]])
        return np.array(wav_output), (intervals/sr*1000).astype(int)

    def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
        linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
        return linear.T


    def load_data(path, win_length=400, 
                sr=16000, hop_length=160, 
                n_fft=512, embedding_per_second=0.5, 
                overlap_rate=0.5):
        
        wav, intervals = load_wav(path, sr=sr)
        linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
        mag, _ = librosa.magphase(linear_spect)  # magnitude
        mag_T = mag.T
        freq, time = mag_T.shape
        spec_mag = mag_T

        spec_len = sr/hop_length/embedding_per_second
        spec_hop_len = spec_len*(1-overlap_rate)

        cur_slide = 0.0
        utterances_spec = []

        while(True):  # slide window.
            if(cur_slide + spec_len > time):
                break
            spec_mag = mag_T[:, int(cur_slide+0.5) : int(cur_slide+spec_len+0.5)]
            
            # preprocessing, subtract mean, divided by time-wise var
            mu = np.mean(spec_mag, 0, keepdims=True)
            std = np.std(spec_mag, 0, keepdims=True)
            spec_mag = (spec_mag - mu) / (std + 1e-5)
            utterances_spec.append(spec_mag)

            cur_slide += spec_hop_len

        return utterances_spec, intervals

    def caculate_der(wav_path, label_path, embedding_per_second=1.0, overlap_rate=0.5):

        specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)

        feats = []
        for spec in specs:
            spec = np.expand_dims(np.expand_dims(spec, 0), -1)
            v = network_eval.predict(spec)
            feats += [v]

        feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]
        
        predicted_label = uisrnnModel.predict(feats, inference_args)
        
        hypothesis = new_utils.result_map(intervals, predicted_label)
        reference = new_utils.reference_rttm(label_path)

        der = new_utils.der(reference, hypothesis)

        new_utils.save_and_export(result_map=hypothesis, 
                                dir=consts.result_dir, 
                                audio_name=wav_path.split("\\")[-1], 
                                der=der['diarization error rate'])

        return der['diarization error rate']
    
    import time
    start = time.time()
    wavs_path = glob.glob("data/test/*.wav")
    labels_path = "data/label_test"
    ders = []
    for i, wav_path in enumerate(wavs_path):
        name_audio = wav_path.split("\\")[-1]
        der = caculate_der(wav_path, labels_path + "\\" + name_audio.replace(".wav", ".rttm"), 
                   embedding_per_second=2, overlap_rate=0.4)
        print("Der of " + wav_path.split("\\")[-1] + ": " + str(der))
        ders.append(der)
    ders = np.array(ders)
    print("Mean DER: " + str(np.mean(ders)))
    end = time.time()
    print("Time to evaluate: " + str(end - start))