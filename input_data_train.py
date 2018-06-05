from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np 
import tensorflow as tf 
import struct
import h5py
import random

from collections import defaultdict
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile

def prepare_audio_settings(sample_rate,
    duration_ms,
    window_size_ms,
    window_stride_ms,
    num_coefficient):
    desired_samples = int(sample_rate * duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    duration_minus_window = desired_samples - window_size_samples
    if duration_minus_window < 0:
        desired_spectrogramme_length = 0
    else:
        desired_spectrogramme_length = 1 + int(duration_minus_window/window_stride_samples)
    return {
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'desired_spectrogramme_length': desired_spectrogramme_length,
        'num_coefficient': num_coefficient,
        'sample_rate': sample_rate
    }

class AudioProcessor(object):
    '''
    1. There are 3 types of text files in data_dir which must be prepared 
        before the initialization of this class: spk2utt, wav.scp, vad.scp

    this class will prepare input audio data for training, the initialization can:
    1. calcule and save mfcc features for all data and give a map: utt-->mfcc
    2. generate file trials for training, each line of that file is a tuple,
        the form:   (utt_evaluation,(utt_enroll1, utt_enroll2, ..., utt_enrollN)) 0/1
                        0:nontarget  1:target
    1. using methode get_data() when we feed a batch of data to our LSTM model. 
    '''  
    def __init__(self, data_dir,
        num_repeats,
        audio_settings,
        skip_generate_feature,
        num_utt_enrollment):
        '''
    the form of data_dir :   ..../train/  or  ...../test/
    the value of is training:  True or False
    the value of skip_generate_feature: True or False
        ''' 
        self.audio_settings = audio_settings
        self.num_utt_enrollment = num_utt_enrollment
        self.generate_trials(data_dir, num_repeats, num_utt_enrollment)
        if not skip_generate_feature:
            self.generate_features(data_dir, audio_settings)
    def read_ark(self, arkpath, offset):
        read_buffer = open(arkpath, 'rb')
        read_buffer.seek(int(offset), 0)
        header = struct.unpack('<xcccc', read_buffer.read(5))
        if header[1] == 'C':
            return 'Error, input .ark file is compressed'
        rows = 0
        _, rows = struct.unpack('<bi', read_buffer.read(5))
        if header[1] == 'F':
            tmp_mat = np.frombuffer(read_buffer.read(rows*4), dtype=np.float32)
        elif header[1] == 'D':
            tmp_mat = np.frombuffer(read_buffer.read(rows*8), dtype=np.float64)
        mat = np.reshape(tmp_mat, (rows))
        read_buffer.close()
        return mat.astype(np.int_)


    def get_vad_from_scp(self, wav_name, scp_filename):
        #wav_name: relative path
        #output the vad array of the input wav_name
        offset =  -1
        read_buffer = open(scp_filename, 'r')
        line = read_buffer.readline()
        while line:
            content = line.split()
            utt_name = content[0]
            if wav_name == utt_name:
                ark_comb = content[1]
                ark_path = ark_comb.split(':')[0]
                offset = int(ark_comb.split(':')[1])
                break
            line = read_buffer.readline()
        read_buffer.close()
        if offset ==  -1:
            print('Error in getting vad, could not find the wav_file: ' + wav_name)
            return 'Error in getting vad'
        else:
            mat_vad = self.read_ark(ark_path, offset)
        return mat_vad
    def get_features_no_sil(self, vad_mat, mfcc_mat):
        #mfcc_mat:   (1,num_frames,num_coeficient)
        #vad_mat:   (num_frames,)
        vadframe_minus_mfccframe = vad_mat.shape[0] - mfcc_mat.shape[1]
        if vadframe_minus_mfccframe >= 0:
            vad_mat = vad_mat[:-vadframe_minus_mfccframe]   # we remove the last two frames of vad_mat
            mfcc_mat = mfcc_mat.reshape((mfcc_mat.shape[1]), mfcc_mat.shape[2])
            list_index = []
            for i,j in enumerate(vad_mat):
                if j == 0:
                    list_index.append(i)
            return np.delete(mfcc_mat, list_index, 0)
        else:
            print('Error frame of vad is less than frame of mfcc')
            return 'Error frame of vad is less than frame of mfcc'
    def generate_features(self, data_dir, audio_settings):
        '''
        read wav.scp vad.scp,and generate  mfcc features for all utts
        wav.scp and vad.scp must be sorted in the same order
        '''
        with tf.Session(graph=tf.Graph()) as sess:
            #calcul mfcc features for utts
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            spectrogram = contrib_audio.audio_spectrogram(wav_decoder.audio,
                window_size=audio_settings['window_size_samples'],
                stride=audio_settings['window_stride_samples'],
                magnitude_squared=True)
            mfcc = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate,
                dct_coefficient_count=audio_settings['num_coefficient'])
            
            wav_scp = data_dir + '/wav.scp'
            vad_scp = data_dir + '/vad.scp'
            read_buffer = open(wav_scp, 'r')
            h5df_filename = data_dir + 'feature_mfcc.h5'
            write_buffer = h5py.File(h5df_filename, 'w')
            for line in read_buffer.readlines():
                content = line.split()
                utt_name = content[0]
                utt_path = content[1]
                vad_mat = self.get_vad_from_scp(utt_name, vad_scp)
                mfcc_mat = sess.run(mfcc, feed_dict={wav_filename_placeholder: utt_path})
                mfcc_mat_no_sil = self.get_features_no_sil(vad_mat, mfcc_mat)   #shape:  (num_frames_no_sil, num_coeficient)
                write_buffer[utt_name] = mfcc_mat_no_sil
            read_buffer.close()
            write_buffer.close()

    def generate_trials(self, data_dir, num_repeats, num_utt_enrollment):
        '''
        1. genenrate trails positive
        2. generate trials negative
        '''
        # generate a dictionnary: spk_id -->  list of all his or her utts
        spk2utt_file = os.path.join(data_dir,'spk2utt')
        read_buffer = open(spk2utt_file, 'r')
        dict_spk2utts = {}
        #construction of dicionary: spk_id --> all his or her utts
        for line in read_buffer.readlines():
            content = line.split()
            spk_id = content[0]
            dict_spk2utts[spk_id] = content[1:]
        read_buffer.close()
        # trials positive
        write_buffer_p = open(os.path.join(data_dir, 'trials_positive'), 'w')
        for i in range(num_repeats):
            for spk in dict_spk2utts.keys():
                utt_samples = random.sample(dict_spk2utts[spk],
                    num_utt_enrollment+1)
                for utt in utt_samples:
                    write_buffer_p.write(utt + ' ')
                write_buffer_p.write('1' + '\n')
        write_buffer_p.close()
        #trials negative
        write_buffer_n = open(os.path.join(data_dir, 'trials_negative'), 'w')
        for spk_eval in dict_spk2utts.keys():
            spk_list = dict_spk2utts.keys()[:]
            spk_list.remove(spk_eval)
            for i in range(num_repeats):
                spk_enroll = random.sample(spk_list, 1)[0]
                utt_eval = random.sample(dict_spk2utts[spk_eval], 1)
                samlpes_utt_enroll = random.sample(dict_spk2utts[spk_enroll],
                    num_utt_enrollment)
                write_buffer_n.write(utt_eval[0] + ' ')
                for utt_enroll in samlpes_utt_enroll:
                    write_buffer_n.write(utt_enroll + ' ')
                write_buffer_n.write('0' + '\n')
        write_buffer_n.close() 

    def get_data(self, trials, read_buffer, label):
        # trials: list of batch_size lines,each line is an tuple
        batch_size = len(trials)
        tuple_size = self.num_utt_enrollment+1
        desired_frames = self.audio_settings['desired_spectrogramme_length']
        feature_size =  self.audio_settings['num_coefficient']
        data = np.zeros((batch_size, tuple_size, desired_frames, feature_size))
        for i,trial in enumerate(trials):
            content = trial.split()
            for j,utt in enumerate(content[:-1]):
                mat_mfcc = read_buffer[utt]
                shape_mfcc = mat_mfcc.shape
                num_frames_mfcc = shape_mfcc[0]
                manque_frames = num_frames_mfcc - desired_frames
                if manque_frames < 0:
                    #padding 0
                    data[i, j] = np.lib.pad(mat_mfcc, ((0, -manque_frames),(0, 0)), 'constant', constant_values=0)
                else: #dont need padding, but we want to chose a start frame randomly
                    start_frame = random.randint(0, manque_frames)
                    data[i, j] = mat_mfcc[start_frame: start_frame+desired_frames]
        return data, label




