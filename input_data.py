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

def prepare_audio_settings(sample_rate, duration_ms, window_size_ms, window_stride_ms, num_coefficient):
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
    1. There are 4 types of text files in data_dir which must be prepared before the initialization of this class: spk2utt, utt2spk , wav.scp, vad.scp

    this class will prepare input audio data for training or testing, the initialization can:
    1. calcule and save mfcc features for all data and give a map: utt-->mfcc
    we divide the datasets into two parts, datasets for training and testing. There are total 252 speakers. 232 for training and 20 for testing
    2. (if we are in training phase)generate a file named 'trials' which gives tuples and labels for training, the form of tuples: (utt1,utt2,utt3....,uttN), the form of labels: 1(target)/0(notarget)
    3. (if we are in testing phase)generate  a file named 'trials' which gives tuples and labels for testing, the form of tuples: (utt1,utt2,utt3....,uttN), labels: 1(target)/0(notarget)
    
    using methode get_data() when we feed a batch of data to our DNN model. 
    '''  
    def __init__(self, data_dir, num_repeats, audio_settings, skip_generate_feature, num_utt_enrollment):
        '''
    num_repeats: the duration of our utts is 2-5s, but the duration for training is 1-2s, we want to full use our utts
    the form of data_dir :   ...../data/train/  or  ....../data/test/
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
        wav.scp and vad.scp are sorted in the same order
        '''
        with tf.Session(graph=tf.Graph()) as sess:
            #calcul mfcc features for utts
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            spectrogram = contrib_audio.audio_spectrogram(wav_decoder.audio, window_size=audio_settings['window_size_samples'], stride=audio_settings['window_stride_samples'], magnitude_squared=True)
            mfcc = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate, dct_coefficient_count=audio_settings['num_coefficient'])
            
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
        create an dictionary:  spk_id --> all of his/her utts
        for each itteration:
            create an enrollment list:   spk_id --> N utts(N=num_utt_enrollment)
            the rest is evaluation list
            pick one utt in evaluation list, and generate N(nomber of  speaker in enrollment list) tuples
            give a label for each tuple using dictionary to know if they are from the same speaker
        '''
        spk2utt_file = data_dir + '/spk2utt'
        read_buffer = open(spk2utt_file, 'r')
        dict_spk2utts = {}
        #construction of dicionary: spk_id --> all his or her utts
        for line in read_buffer.readlines():
            content = line.split()
            spk_id = content[0]
            dict_spk2utts[spk_id] = content[1:]
        read_buffer.close()
        # generate the file trials in the form:    utt_evaluation utt_enrollment1 utt_enrollment2 .. utt_enrollmentN
        list_spk = dict_spk2utts.keys()
        trials_file_name = data_dir + '/trials'
        write_buffer = open(trials_file_name, 'w')
        for iteration in range(num_repeats):
            dict_spk2utts_enrollment = {}
            utt_list_evaluation = []
            for spk in list_spk:
                utt_list = list(dict_spk2utts[spk])
                num_utt = len(utt_list)
                if num_utt <= num_utt_enrollment:
                    utt_list_evaluation += utt_list
                    continue
                list_utt_enrollment = random.sample(utt_list, num_utt_enrollment)
                dict_spk2utts_enrollment[spk] = list_utt_enrollment
                for element in list_utt_enrollment:
                    utt_list.remove(element)
                utt_list_evaluation += utt_list
            random.shuffle(utt_list_evaluation)

            for utt_evalutation in utt_list_evaluation:
                for enroll_spk in dict_spk2utts_enrollment.keys():
                    write_buffer.write(utt_evalutation)
                    for utt_enrollment in dict_spk2utts_enrollment[enroll_spk]:
                        write_buffer.write(' ' + utt_enrollment)
                    if utt_evalutation in dict_spk2utts[enroll_spk]:  #target, we note 1
                        write_buffer.write(' 1'+'\n')
                    else:#non target , we note 0
                        write_buffer.write(' 0'+'\n')
        write_buffer.close()

    def get_data(self, one_trial, read_buffer):
        # if number of frames of mfcc is less than than duration, pad zeros
        content = one_trial.split()
        label = int(content[-1])
        content.remove(content[-1])
        tuple_size = self.num_utt_enrollment + 1
        feature_size = self.audio_settings['num_coefficient']
        desired_frames = self.audio_settings['desired_spectrogramme_length']
        voice_print = tf.zeros([1, feature_size])
        for utt in content:
            mfcc_feature = read_buffer[utt]    #shape:  (num_frames_no_sil, num_coeficient)
            feature_shape = mfcc_feature.shape
            num_frames = feature_shape[0]
            minus = num_frames - desired_frames
            if minus < 0:  #need padding
                mfcc_feature = tf.pad(mfcc_feature, [[0, -minus],[0, 0]])
            else:  #dont need padding, but we want to chose a start frame randomly
                start_frame = random.randint(0, minus)
                mfcc_feature = mfcc_feature[start_frame:start_frame+desired_frames]
            voice_print = tf.concat([voice_print, mfcc_feature], 0)
        voice_print = voice_print[1:]   #shape: (tuple_size*desired_speclength,feature_size)
        voice_print = tf.reshape(voice_print, [tuple_size, desired_frames, feature_size])
        with tf.Session() as sess:
            voice_print = sess.run(voice_print)
        return voice_print, label




