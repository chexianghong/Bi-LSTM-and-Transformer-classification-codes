# Dec 29, 2020
# model builder
# import model_build


# refer to
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/quickstart/advanced

import math
import numpy as np
import logging
import pandas as pd

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import Input, Model
import tensorflow_addons as tfa
from keras import backend as K
from tensorflow.keras.layers import Dropout,Softmax
from tensorflow.keras.layers import LayerNormalization, MaxPooling1D, AveragePooling1D,Conv1D
from tensorflow.keras.layers import Masking,Embedding
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
from tensorflow.keras import Input, Model
import tensorflow_addons as tfa
from keras.callbacks import TensorBoard, EarlyStopping
from typing import Callable, Union

N_times = 14
N_feature = 14
N_outputs = 8
batch_size = 32


class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr

        logs.update({'lr': K.eval(current_lr)})
        super().on_epoch_end(epoch, logs)


## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, target_lr=1e-3, start_lr=0.0):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step_tf = tf.cast(global_step, tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(
        tf.constant(np.pi) * (global_step_tf - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold, learning_rate, target_lr)

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return tf.cast(learning_rate, tf.float32)

## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=step, total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr, hold=self.hold)

        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")

    def get_config(self):
        config = {
            'start_lr': self.start_lr,
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'hold': self.hold}
        return config

## **************************************
def embedding_f(inputx, layer_n=3, unit=128):
    he = tf.keras.initializers.HeNormal()
    
    x = inputx
    for i in range(layer_n):
        # x = Dense(units=unit//2**(layer_n-i-1),activation="relu", kernel_initializer=he)(x)
        x = Dense(units=unit, activation="relu", kernel_initializer=he)(x)
        x = BatchNormalization()(x)
    
    return x


## **************************************
def decoder_f(inputx, layer_n=3, unit=128):
    he = tf.keras.initializers.HeNormal()
    
    x = inputx
    for i in range(layer_n):
        # x = Dense(units=unit//2**i,activation="relu", kernel_initializer=he)(x)
        x = Dense(units=unit, activation="relu", kernel_initializer=he)(x)
        x = BatchNormalization()(x)
    
    return x


def positional_encoding(length, depth):
    depth = depth // 2
    
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff, reg=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=reg),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_regularizer=reg)  # (batch_size, seq_len, d_model)
    ])

## check test_mask.py to see why this is like adding two new axises
def create_padding_mask(inputs, mask_value=0):
    seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# layern=1; units=64; n_times=95; n_times_out=95; n_feature=2; n_head=8; is_batch=False; drop=0; mask_value = 3.2767;  active="linear"
# inputs = X_rnn_train[:16,:,:] # no filled data
# inputs = [train_x_ran[:16,:,:],train_y_ran[:16,:].reshape(16,-1,1)] # no filled data
# https://www.tensorflow.org/text/tutorials/transformer


def get_transformer(layern=3, units=124, n_times=14, n_feature=2, n_head=4, drop=0.1, n_out=9, is_batch=True, mask_value=-9999.0, active="exponential"):
    gu = tf.keras.initializers.GlorotUniform()
    gn = tf.keras.initializers.GlorotNormal()
    he = tf.keras.initializers.HeNormal()
    inputs = Input(shape=(n_times, n_feature,))
    # inputs = [Input(shape=(n_times,n_feature,)),Input(shape=(n_times_out,n_feature,))]
    # return_sequences = sequence if layern<2 else True
    
    embedding = Dense(units, activation="relu")
    embedding_p = Dense(units, activation="relu")
    ## *******************
    # positional
    from keras import backend as K
    b = K.ones_like(inputs[:, :, :1])
    xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
    xpp = b * xp
    # xpp = tf.identity(inputs[:,:,:1] )
    # xpp[0,:,:] = xp
    # x = Masking(mask_value=mask_value, input_shape=(n_times, n_feature)) (inputs)
    x = inputs
    # mask_multi = tf.cast(x != mask_value, tf.float32)
    mask_multi = tf.cast(tf.math.not_equal(x,mask_value), tf.float32)
    x = x * mask_multi
    x = embedding(x)
    x = x + embedding_p(xpp)
    
    padding_mask = create_padding_mask(inputs=inputs, mask_value=mask_value)

    # encoder
    for i in range(layern):
        # temp_mha = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head)
        # attn_output, attn4 = temp_mha(query=x,value=x,key=x,return_attention_scores=True, attention_mask=padding_mask)
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head)(query=x, value=x, key=x,
                                                                                           return_attention_scores=True,
                                                                                           attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        
        ffn_output = point_wise_feed_forward_network(units, units * 4)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        
        x = out2
    
    # enc_output = x
    # enc_output = AveragePooling1D(pool_size=n_times)(enc_output)
    # enc_output = tf.reshape(enc_output,[-1,enc_output.shape[2]])
    # output = Dense(n_out, activation=active)(enc_output)
    # model = Model(inputs, output)

    enc_output = x
    enc_output1 = MaxPooling1D(pool_size=n_times)(enc_output)
    enc_output2 = AveragePooling1D(pool_size=n_times)(enc_output)
    enc_output = enc_output1 + enc_output2
    enc_output = tf.reshape(enc_output, [-1, enc_output.shape[2]])
    output = Dense(n_out, activation='softmax')(enc_output)
    model = Model(inputs, output)

    # enc_output = x
    # enc_output1 = MaxPooling1D(pool_size=n_times)(enc_output)
    # enc_output2 = AveragePooling1D(pool_size=n_times)(enc_output)
    # enc_output = tf.concat([enc_output1, enc_output2], 2)
    # enc_output = tf.reshape(enc_output, [-1, enc_output.shape[2]])
    # output = Dense(n_out, activation=active)(enc_output)
    # model = Model(inputs, output)
    
    return model


def get_transformer_new_att0(n_times=14, n_feature=2, n_out=9, layern=3, units=64, n_head=4, drop=0.1, is_batch=True,
                             mask_value=-9999.0, active="softmax", is_att=False, L2=0):
    """using AveragePooling1D with mask"""
    inputs = Input(shape=(n_times, n_feature,))
    embedding_x = Dense(units)
    embedding_p = Dense(units)
    reg = None
    if L2 > 0:
        reg = tf.keras.regularizers.l2(l=L2)

    ## *******************
    # positional
    b = K.ones_like(inputs[:, :, :1])
    xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
    xpp = b * xp

    x0 = inputs

    mask_multi = tf.cast(tf.math.not_equal(x0, mask_value), tf.float32)
    x0 = x0 * mask_multi
    # x = embedding_x(x)
    x = embedding_x(x0) + embedding_p(xpp)
    # xx = tf.cast(tf.math.not_equal(x,mask_value), tf.bool)

    padding_mask = create_padding_mask(inputs=inputs, mask_value=mask_value)
    # encoder
    for i in range(layern):
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head, kernel_regularizer=reg)(
            query=x, value=x, key=x,
            return_attention_scores=True,
            attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)

        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)

        ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)

        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)

        x = out2

    enc_output = x
    enc_output2 = tf.math.multiply(enc_output, mask_multi[:, :, :1])
    # msk0 = tf.zeros_like(enc_output, dtype=tf.float32)
    # enc_output = tf.where(xx, enc_output, msk0)
    # enc_output1 = MaxPooling1D(pool_size=n_times)(enc_output)
    # enc_output2 = AveragePooling1D(pool_size=n_times)(enc_output)
    # enc_output2 =  K.sum(enc_output2, axis=1) / K.sum(mask_multi[:,:,:1], axis=1)
    enc_output2 = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(mask_multi[:, :, :1], axis=1))
    if is_att:
        enc_output2 = enc_output2[:, tf.newaxis, ]
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head, kernel_regularizer=reg)(
            query=enc_output2, value=x, key=x, return_attention_scores=True,
            attention_mask=padding_mask)
        # enc_output = enc_output1 + enc_output2
        enc_output = attn_output
        enc_output = tf.reshape(enc_output, [-1, enc_output.shape[2]])
    else:
        enc_output = enc_output2

    output = Dense(n_out, activation=active, kernel_regularizer=reg)(enc_output)
    model = Model(inputs, output)
    return model

def trainings_val_class(train_ds, validation_ds, dd_model, ff_hist, num_epochs, lr, _num_s):
    #
    # model = get_transformer(layern=3, units=128, n_times = N_times, n_feature = N_feature, n_head=4, drop=0.5, n_out=N_outputs,
    #                  is_batch=True, mask_value=-9999.0, active="exponential")

    model = get_transformer_new_att0(n_times=N_times, n_feature=N_feature, n_out=N_outputs, layern=3, units=64, n_head=4, drop=0.1, is_batch=True,
                             mask_value=-9999.0, active="softmax", is_att=False, L2=0.0001)

    # early_stopping = tf.keras.callbacks.EarlyStopping('val_sparse_categorical_accuracy', patience=5,restore_best_weights=True)
    # save_best = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=dd_model,
    #     monitor='val_sparse_categorical_accuracy',
    #     save_best_only=True,
    #     mode='max',
    #     verbose=1
    # )

    #
    # per_epoch = _num_s // batch_size
    # split_epoch = 20
    # momentum = 0.9
    # start_rate = 0.001
    # warm_up = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=start_rate / split_epoch / per_epoch,
    #                                                         decay_steps=split_epoch * per_epoch,
    #                                                         end_learning_rate=start_rate, name='Decay_linear')
    # optimizer = tf.keras.optimizers.Adam(learning_rate=warm_up, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    #
    # decay = 1e-5
    # optimizer = tfa.optimizers.AdamW(weight_decay=decay, learning_rate=0.0001, beta_1=0.9)
    # model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    #
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    hist = model.fit(train_ds, validation_data=validation_ds, epochs=num_epochs, verbose=1)
    model.save(dd_model)

    hist_df = pd.DataFrame(hist.history)
    with open(ff_hist, mode='w') as f:
        hist_df.to_csv(f)

