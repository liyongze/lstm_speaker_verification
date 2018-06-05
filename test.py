from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import argparse
import sys
import numpy as np 
import tensorflow as tf 
import input_data_test
import model
import h5py

from tensorflow.python.platform import gfile

FLAGS = None
def score(batch_size, tuple_size, spk_rep, batch_label):
    feature_size = spk_rep.shape[1]
    w =  tf.reshape(spk_rep, [batch_size, tuple_size, feature_size])
    score_batch = tf.zeros([1])
    for indice_bash in range(batch_size):
        wi_enroll = w[indice_bash, 1:]    # shape:  (tuple_size-1, feature_size)
        wi_eval = w[indice_bash, 0]
        normlize_wi_enroll = tf.nn.l2_normalize(wi_enroll, dim=1)
        c_k = tf.reduce_mean(normlize_wi_enroll, 0)              # shape: (feature_size)
        normlize_ck = tf.nn.l2_normalize(c_k, dim=0)
        normlize_wi_eval = tf.nn.l2_normalize(wi_eval, dim=0)
        cos_similarity = tf.reduce_sum(tf.multiply(normlize_ck,normlize_wi_eval))
        score = cos_similarity
        score_batch = tf.concat([score_batch, [score]], 0)
    score_batch = score_batch[1:]
    return score_batch

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    #generate  an dictionary which indicate some settings of traitement audio
    audio_settings = input_data_test.prepare_audio_settings(
        FLAGS.sample_rate,
        FLAGS.duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.num_coefficient)
    # if skip_generate_feature=True, it will calculs the mfcc feature and prepare the file trials for training or testing
    audio_data_processor = input_data_test.AudioProcessor(
        FLAGS.data_dir,
        FLAGS.num_repeats,
        audio_settings,
        FLAGS.skip_generate_feature,
        FLAGS.num_utt_enrollment)
    # create an objet of class AudioProcessor. 
    lstm_model_setting={}
    lstm_model_setting['num_units'] = FLAGS.num_units
    lstm_model_setting['dimension_projection'] = FLAGS.dimension_projection
    lstm_model_setting['num_layers'] = FLAGS.num_layers
    #creat graph
    input_audio_data = tf.placeholder(tf.float32,
        [FLAGS.batch_size, 1+FLAGS.num_utt_enrollment, audio_settings['desired_spectrogramme_length'], FLAGS.num_coefficient],
        name='input_audio_data')   
    dimension_linear_layer = FLAGS.dimension_linear_layer
    weights = tf.Variable(tf.random_normal([FLAGS.dimension_projection,dimension_linear_layer], stddev=1), name='weights')
    bias = tf.Variable(tf.random_normal([dimension_linear_layer], stddev=1), name='bias')
    #weight_scalar = tf.Variable(1.0, name='weight_scalar')
    #bias_scalar = tf.Variable(0.1, name='bias_scalar')
    dropout_prob_input = tf.placeholder(tf.float32, [], name='dropout_prob_input')
    #  output of the model
    outputs = model.create_lstm_baseline_model(
        audio_tuple_input= input_audio_data, 
        W=weights,
        B=bias,
        lstm_model_setting=lstm_model_setting,
        dropout_prob=dropout_prob_input)
 
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.file_checkpoint))


    batch_size = FLAGS.batch_size
    read_mfcc_buffer = h5py.File(FLAGS.data_dir + '/feature_mfcc.h5', 'r')
    read_trials  = open(FLAGS.data_dir + '/trials', 'r')

    trials = read_trials.readlines()
    num_iteration = int(len(trials)/batch_size)
    #calcul score of a batch
    batch_label = tf.placeholder(tf.int32, [batch_size])
    batch_score = score(batch_size=batch_size, tuple_size=FLAGS.num_utt_enrollment+1, spk_rep=outputs, batch_label=batch_label)
    file_score = open(os.path.join(FLAGS.data_dir, 'score_eval'), 'w')
    for i in range(num_iteration):
        trial_batch = trials[i*batch_size: (i+1)*batch_size]
        test_voiceprint, label = audio_data_processor.get_data(trial_batch, read_mfcc_buffer)
        score_batch = sess.run(batch_score, feed_dict={input_audio_data:test_voiceprint, dropout_prob_input:0, batch_label:label})
        #shape of output:  batch_size*tuple_size
        #shape of score_batch: batch_size
        for i in range(score_batch.shape[0]):
            file_score.write(str(score_batch[i]) + '\n')
    read_mfcc_buffer.close()
    read_trials.close()
    file_score.close()


if __name__ == '__main__':
    pwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='sample rate of the wavs'
    )
    parser.add_argument(
        '--duration_ms',
        type=int,
        default=2000,
        help='duration of wavs used for training'
        )
    parser.add_argument(
        '--window_size_ms',
        type=int,
        default=25,
        help='how long each frame of spectrograme'
        )
    parser.add_argument(
        '--window_stride_ms',
        type=int,
        default=10,
        help='how far to move in time between two frames'
        )
    parser.add_argument(
        '--num_coefficient',
        type=int,
        default=40,
        help='numbers of coefficients of mfcc'
        )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/export/maryland/yzli/lstm_speaker_verification/test/',
        help='work location'
        )
    parser.add_argument(
        '--num_repeats',
        type=int,
        default=140,
        help='number of repeat when we prepare the trials'
        )
    parser.add_argument(
        '--skip_generate_feature',
        type=bool,
        default=True,
        help='whether to skip the phase of generating mfcc features'
        )
    parser.add_argument(
        '--num_utt_enrollment',
        type=int,
        default=5,
        help='numbers of enrollment utts for each speaker'
        )
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=True,
        help='whether to check for invalid numbers during processing'
        )
    parser.add_argument(
        '--num_units',
        type=int,
        default=128,
        help='numbers of units for each layer of lstm'
        )
    parser.add_argument(
        '--dimension_projection',
        type=int,
        default=64,
        help='dimension of projection layer of lstm'
        )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='number of layers of multi-lstm'
        )
    parser.add_argument(
        '--dimension_linear_layer',
        type=int,
        default=64,
        help='dimension of linear layer on top of lstm'
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=80
        )
    parser.add_argument(
        '--file_checkpoint',
        type=str,
        default='../train/'
        )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)