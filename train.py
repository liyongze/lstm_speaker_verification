from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import argparse
import sys
import numpy as np 
import tensorflow as tf 
import input_data
import model
import h5py

from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf.InteractiveSession()
    #generate  an dictionary which indicate some settings of traitement audio
    audio_settings = input_data.prepare_audio_settings(
        FLAGS.sample_rate,
        FLAGS.duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.num_coefficient)
    # create an objet of class AudioProcessor. 
    lstm_model_setting={}
    lstm_model_setting['num_units'] = FLAGS.num_units
    lstm_model_setting['dimension_projection'] = FLAGS.dimension_projection
    lstm_model_setting['num_layers'] = FLAGS.num_layers
    # if skip_generate_feature=True, it will calculs the mfcc feature and prepare the file trials for training or testing
    audio_data_processor = input_data.AudioProcessor(
        FLAGS.data_dir,
        FLAGS.num_repeats,
        audio_settings,
        FLAGS.skip_generate_feature,
        FLAGS.num_utt_enrollment)
    #hold a place for input of the neural network
    input_audio_data = tf.placeholder(tf.float32,
        [1+FLAGS.num_utt_enrollment, audio_settings['desired_spectrogramme_length'], FLAGS.num_coefficient],
        name='input_audio_data')
    # definit weight W and bias B for the linear layer on top of the LSTM
    dimension_linear_layer = FLAGS.dimension_linear_layer
    weights = tf.Variable(tf.random_normal([FLAGS.dimension_projection,dimension_linear_layer], stddev=1), name='weights')
    bias = tf.Variable(tf.random_normal([dimension_linear_layer], stddev=1), name='bias')
    dropout_prob_input = tf.placeholder(tf.float32, [], name='dropout_prob_input')
    #  output of the model
    if FLAGS.model_architechture == 'lstm_baseline':
        outputs = model.create_lstm_baseline_model(
        audio_tuple_input= input_audio_data, 
        W=weights,
        B=bias,
        lstm_model_setting=lstm_model_setting,
        dropout_prob=dropout_prob_input)
    # hold the place for label: 0:nontarget  1:target
    labels = tf.placeholder(tf.int64, [], name='labels')
    # check Nan or other numeriical errors
    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]
    #get the loss 
    weight_scalar = tf.Variable(1.0, name='weight_scalar')
    bias_scalar = tf.Variable(0.1, name='bias_scalar')
    with tf.name_scope('train_loss'):
        loss = model.tuple_loss(spk_representation=outputs, labels=labels, weight=weight_scalar, bias=bias_scalar)

    tf.summary.scalar('train_loss', loss)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(tf.float32, name='learning_rate_input')
        train_step  = tf.train.AdamOptimizer(learning_rate=learning_rate_input).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())

    #merge all the summaries
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.data_dir + '/logs', sess.graph)
    # training loop
    tf.global_variables_initializer().run()
    training_step = 0
    read_mfcc_buffer = h5py.File(FLAGS.data_dir + '/feature_mfcc.h5', 'r')
    read_trials = open(FLAGS.data_dir + '/trials','r')
    all_trials = read_trials.readlines()
    max_training_step = len(all_trials)
    tf.logging.info('total steps %d: ', max_training_step)
    for one_trial in all_trials: 
        training_step += 1
        train_voiceprint, label = audio_data_processor.get_data(one_trial, read_mfcc_buffer)   # get one tuple for training
        #shape of train_voiceprint: (tuple_size, feature_size)    
        #shape of  label:  (1)
        train_summary, train_loss, _ = sess.run([merged_summaries, loss, train_step],
            feed_dict={input_audio_data:train_voiceprint, labels:label, learning_rate_input:FLAGS.learning_rate, dropout_prob_input:FLAGS.dropout_prob})
        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: loss %f' %(training_step, train_loss))

        #save  the model
        if training_step == max_training_step:
            save_path = os.path.join(FLAGS.data_dir, FLAGS.model_architechture + '.ckpt')
            tf.logging.info('saving to "%s-%d"', save_path, training_step)
            saver.save(sess, save_path, global_step=training_step)
    read_mfcc_buffer.close()
    read_trials.close()
    

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
        default=1000,
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
        default=pwd,
        help='work location'
        )
    parser.add_argument(
        '--num_repeats',
        type=int,
        default=1,
        help='number of repeat when we prepare the trials'
        )
    parser.add_argument(
        '--skip_generate_feature',
        type=bool,
        default=None,
        help='whether to skip the phase of generating mfcc features'
        )
    parser.add_argument(
        '--num_utt_enrollment',
        type=int,
        default=2,
        help='numbers of enrollment utts for each speaker'
        )
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=True,
        help='whether to check for invalid numbers during processing'
        )
    parser.add_argument(
        '--model_architechture',
        type=str,
        default='lstm_baseline'
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
        default=3,
        help='number of layers of multi-lstm'
        )
    parser.add_argument(
        '--dimension_linear_layer',
        type=int,
        default=64,
        help='dimension of linear layer on top of lstm'
        )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001
        )
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.5
        )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)