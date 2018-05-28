from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
def unit_lstm(num_units, dimension_projection, dropout_prob):
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, num_proj=dimension_projection, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout_prob)
    return lstm_cell

def create_lstm_baseline_model(audio_tuple_input, W, B, lstm_model_setting, dropout_prob):
    '''
    audio_tuple_input:  shape: (tuple_size, time_steps, feature_size)
    '''

    #reshape_input_data
    tuple_size = audio_tuple_input.shape[0]
    time_steps = audio_tuple_input.shape[1]
    feature_size = audio_tuple_input.shape[2]
    XT = tf.transpose(audio_tuple_input, [1, 0, 2]) #shape (time_steps,tuple_size, feature_size)
    XR = tf.reshape(XT, [-1, feature_size])  #shape: (time_steps*tuple_size, feature_size)
    X_split = tf.split(XR, time_steps, 0)  # X_split has timesteps arrays ,and each array has dimension(tuple_size, feature_size)

    #definit lstm
    num_units = lstm_model_setting['num_units']
    dimension_projection = lstm_model_setting['dimension_projection']
    num_layers = lstm_model_setting['num_layers']
    mult_lstm = tf.nn.rnn_cell.MultiRNNCell([unit_lstm(num_units, dimension_projection, dropout_prob) for i in range(num_layers)], state_is_tuple=True)
    # get output of lstm
    outputs, _states = tf.contrib.rnn.static_rnn(mult_lstm, X_split, dtype=tf.float32)
    return tf.matmul(outputs[-1], W) + B


def tuple_loss(spk_representation, labels, weight, bias):
    '''
    this function can calcul the tuple loss for a tuple
    so shape of spk_representation: (tuple_size, dimension of linear layer)
    labels:      shape(1)
    '''
    feature_size = spk_representation.shape[1]
    w_j = spk_representation[0]   #d-vector of evaluation utt
    w_k =  spk_representation[1:] #d-vector of enrollment utt

    normlize_wk = tf.nn.l2_normalize(w_k, dim=1)
    c_k = tf.reduce_mean(normlize_wk, 0)   #shape: (feature_size)
    normlize_ck = tf.nn.l2_normalize(c_k, dim=0)
    normlize_wj = tf.nn.l2_normalize(w_j, dim=0)
    cos_similarity = tf.reduce_sum(tf.multiply(normlize_ck, normlize_wj))
    score = weight * cos_similarity + bias 
    if labels == 1:   #target
        tuple_loss = tf.sigmoid(score)
    else: # nontarget
        tuple_loss = 1 - tf.sigmoid(score)
    return -tf.log(tuple_loss)
