import logging
import os
import numpy as np
import math
import tensorflow as tf
import keras_tuner as kt
import keras
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras import Input,Model
from keras import backend as K

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization,LayerNormalization
from tensorflow.keras.layers import LSTM,GRU,Bidirectional

from tensorflow.keras.layers import Dropout,Softmax
from tensorflow.keras.layers import LayerNormalization, MaxPooling1D, AveragePooling1D,Conv1D
from tensorflow.keras.layers import Masking,Embedding
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
from tensorflow.keras import Input, Model
import tensorflow_addons as tfa
from keras.callbacks import TensorBoard, EarlyStopping
from typing import Callable, Union

# tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)


# N_times = 29
N_times = 14
N_feature = 14
# N_outputs = 8


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

def model_Bidirectional_or_GRU_mask(n_outputs, layern=3, units=45, n_times=24, n_feature=14, ReLU=False, drop=0.5, sequence=False,
                                    is_batch=False, mask=True, active="linear"):
    rnn = Sequential()

    ## get the elements number in each unit
    if isinstance(units, int):
        units_array = np.repeat(units, 100)
    else:
        units_array = np.array(units)

    ## set
    gn = tf.keras.initializers.GlorotNormal()
    he = tf.keras.initializers.HeNormal()
    # gu = tf.keras.initializers.GlorotUniform() if not ReLU else he
    gu = tf.keras.initializers.GlorotNormal() if not ReLU else he
    active_gru = 'tanh' if not ReLU else "relu"
    # Adding our first LSTM layer
    if mask:
        rnn.add(Masking(mask_value=-9999, input_shape=(n_times, n_feature)))
        # rnn.add(GRU(units=units_array[0], activation=active_gru, kernel_initializer=gu, return_sequences=True))
        rnn.add(
            Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences=True), merge_mode='concat'))
    else:
        # rnn.add(GRU(units=units_array[0], activation=active_gru, kernel_initializer=gu, return_sequences=True,
        #           input_shape=(n_times, n_feature)))
        rnn.add(Bidirectional(
            LSTM(units=units_array[0], kernel_initializer=gn, return_sequences=True, input_shape=(n_times, n_feature)),
            merge_mode='concat'))

    # Perform some dropout regularization
    if drop > 0:
        rnn.add(Dropout(drop))

    if is_batch == True:
        rnn.add(LayerNormalization(epsilon=1e-6))

    # Adding three more LSTM layers with dropout regularization
    # for i in [True, True, False]:
    for i in range(layern):
        return_sequences = True
        return_sequences = True if i < (layern - 1) else sequence
        # rnn.add(LSTM(units=units_array[i+1], kernel_initializer=gn, return_sequences=return_sequences))
        rnn.add(Bidirectional(LSTM(units=units_array[i + 1], kernel_initializer=gn, return_sequences=return_sequences),
                              merge_mode='concat'))

        if drop > 0:
            rnn.add(Dropout(drop))
        if is_batch == True:
            rnn.add(LayerNormalization(epsilon=1e-6))

    rnn.add(Dense(100, activation='relu'))
    rnn.add(BatchNormalization())
    rnn.add(Dense(n_outputs, activation='softmax'))
    return rnn

def model_Bidirectional_or_GRU_attention_mask(n_outputs, layern=3, units=45, n_times=24, n_feature=14, ReLU=False, drop=0.5, sequence=False,
                                    is_batch=False, mask=True, active="linear"):
    n_times = N_times
    n_feature = N_feature
    n_outputs = N_outputs
    gn = tf.keras.initializers.GlorotNormal()
    
    if isinstance(units, int):
        units_array = np.repeat(units, 100)
    else:
        units_array = np.array(units)
    
    inputs = Input(shape=(n_times, n_feature,))
    if mask:
        x = Masking(mask_value=-9999, input_shape=(n_times, n_feature))(inputs)
        x = Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences=True, return_state=False), merge_mode='concat')(x)
    else:
        x = Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences=True, return_state=False),merge_mode='concat')(inputs)
 
    if drop > 0:
        x = Dropout(0.5)(x)

    if is_batch == True:
        x = LayerNormalization(epsilon=1e-6)(x)
   
    for i in range(2):
        x = Bidirectional(LSTM(units=units_array[i+1], kernel_initializer=gn, return_sequences=True, return_state=False), merge_mode='concat')(x)
        x = Dropout(0.5)(x)
        if is_batch == True:
            x = LayerNormalization(epsilon=1e-6)(x)

    lstmx = Bidirectional(LSTM(units=units_array[3], kernel_initializer=gn, return_sequences=False, return_state=False),merge_mode='concat')(x)
    lstmx = Dropout(0.5)(lstmx)
    lstmx = LayerNormalization(epsilon=1e-6)(lstmx)

    # NOTE if hre lastmx is [batchsize, dim] -> you need to covert it to [batchsize, 1, dim] by doing below
    lstmx_t = lstmx[:, tf.newaxis, :]
    n_head = 8
    units_last = lstmx.shape[-1]
    
    if mask:
        inputmasking = tf.cast(tf.math.not_equal(inputs[:, :, 0], -9999), tf.float32)
        inputmasking2 = inputmasking[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

        # NOTE LET me know if below has error
        attn_output_x, attn = MultiHeadAttention(key_dim=units_last // n_head, num_heads=n_head)(query=lstmx_t, value=x,
                                                                                                 key=x,
                                                                                             return_attention_scores=True,
                                                                                     attention_mask=inputmasking2)
    else:
        attn_output_x, attn = MultiHeadAttention(key_dim=units_last // n_head, num_heads=n_head)(query=lstmx_t, value=x,
                                                                                                 key=x,
                                                                                               return_attention_scores=True)

    x4 = Concatenate()([lstmx, tf.reshape(attn_output_x, [-1, units_last])])
    # x4 = Concatenate()([lstmx_t, attn_output_x])

    y = Dense(100, activation='relu')(x4)
    y = BatchNormalization()(y)
    y = Dense(n_outputs, activation='softmax')(y)
    model = Model(inputs, y)
    return model


PLAT_N = 20
PLAT_N = 30
# PLAT_N = 40
VALIDATION_STEP = 50
# BATCH = 128
# # LEARNING_RATE = 0.01
# # EPOCH = 50



# @tf.function
def is_validation_flat(validation_accuracies):
    n = len(validation_accuracies)
    sub_n = math.ceil(PLAT_N / 2)
    last_half = validation_accuracies[(n - sub_n):n]
    penultimate_half = validation_accuracies[(n - 2 * sub_n):(n - sub_n)]
    return np.array(penultimate_half).mean() < np.array(last_half).mean()


@tf.function
def loss_reg(model, x, y, training):
    y_ = tf.reshape(model(x, training=training),y.shape ) ## a bug on Dec 29 2020 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mse = tf.keras.losses.MSE(y_true=y, y_pred=y_)
    return tf.math.sqrt(tf.math.reduce_mean(mse)) ## changed on Dec 31 2020 & fixed

@tf.function
def loss_class(model, x, y, training):
    y_ = model(x, training=training)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return loss_object(y_true=y, y_pred=y_)

@tf.function
def grad(model, inputs, targets, reg=True, mask_value=math.nan):
    with tf.GradientTape() as tape:
        if reg:
            if math.isnan(mask_value):
                loss_value = loss_reg(model, inputs, targets, training=True)
            else:
                y_ = tf.reshape(model(inputs, training=True), targets.shape)
                mask = tf.cast(targets != mask_value, y_.dtype)
                targets = tf.cast(targets, y_.dtype)
                abs_vec = tf.multiply(tf.math.square(tf.abs(y_ - targets)), mask)
                loss_value = tf.math.sqrt(tf.reduce_sum(abs_vec) / tf.reduce_sum(mask))
        else:
            loss_value = loss_class(model, inputs, targets, training=True)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)



def trainings_val_class1(train_ds, validation_ds, dd_model, ff_hist, num_epochs, _attention, lr, mask=True):
    if _attention:
        model = model_Bidirectional_or_GRU_attention_mask(N_outputs, 3, [32, 64, 128, 256], N_times, N_feature, False, 0.5,
                                                          False,
                                                          False, mask, "linear")
    else:
        model = model_Bidirectional_or_GRU_mask(N_outputs, 3, [32, 64, 128, 256], N_times, N_feature, False, 0.5, False,
                                                False, mask, "linear")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    hist = model.fit(train_ds, validation_data=validation_ds, epochs=num_epochs, verbose=1)
    model.save(dd_model)

    hist_df = pd.DataFrame(hist.history)
    with open(ff_hist, mode='w') as f:
        hist_df.to_csv(f)

def model_Bidirectional_or_GRU_mask1(n_outputs, layern=3, units=45, n_times=24, n_feature=14, ReLU=False, drop=0.5, sequence=False,
                                    is_batch=False, mask=True, active="linear"):
    gn = tf.keras.initializers.GlorotNormal()
    he = tf.keras.initializers.HeNormal()
    gu = tf.keras.initializers.GlorotNormal() if not ReLU else he
    active_gru = 'tanh' if not ReLU else "relu"

    if isinstance(units, int):
        units_array = np.repeat(units, 100)
    else:
        units_array = np.array(units)

    inputs = Input(shape=(n_times, n_feature,))
    if mask:
        x = Masking(mask_value=-9999, input_shape=(n_times, n_feature))(inputs)
        x = Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences=True),merge_mode='concat')(x)

    else:
        x = Bidirectional(LSTM(units=units_array[0], kernel_initializer=gn, return_sequences=True),merge_mode='concat')(inputs)

    if drop > 0:
        x = Dropout(0.5)(x)

    if is_batch == True:
        x = LayerNormalization(epsilon=1e-6)(x)

    for i in range(layern):
        return_sequences = True
        return_sequences = True if i < (layern - 1) else sequence
        x = Bidirectional(LSTM(units=units_array[i + 1], kernel_initializer=gn, return_sequences=return_sequences), merge_mode='concat')(x)

        if drop > 0:
            x = Dropout(0.5)(x)
        if is_batch == True:
            x = LayerNormalization(epsilon=1e-6)(x)

    y = Dense(100, activation='relu')(x)
    y = BatchNormalization()(y)
    y = Dense(n_outputs, activation='softmax')(y)
    model = Model(inputs, y)
    return model


def trainings_val_class2(N_outputs, train_num, batch_size, train_ds, validation_ds, dd_model, ff_hist, epochs, _attention, lr, mask=True):

    model = model_Bidirectional_or_GRU_mask1(N_outputs, 3, [32, 64, 128, 256], N_times, N_feature, False, 0.5, False,
                                                False, mask, "linear")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    per_epoch = train_num // batch_size

    validation_split = 0
    split_epoch = 5
    hold_epoch = 0
    L2 = 1e-5
    option = 1
    reduce_epoch = False
    momentum = 0.9

    total_steps = per_epoch * epochs
    warmup_steps = per_epoch * split_epoch
    hold_steps = per_epoch * hold_epoch
    schedule_one = WarmUpCosineDecay(start_lr=0.0, target_lr=lr, warmup_steps=warmup_steps,
                                     total_steps=total_steps, hold=hold_steps)

    if option == 0:
        print('tfa.optimizers.RMSprop ');
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=schedule_one, rho=momentum);
    elif option == 1:
        print('tfa.optimizers.Adam ');
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule_one, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        print('tfa.optimizers.AdamW ');
        optimizer = tfa.optimizers.AdamW(weight_decay=decay, learning_rate=schedule_one, beta_1=momentum)

    # model.compile(optimizer=optimizer, loss=loss,  metrics=['accuracy'])
    model.compile(optimizer=optimizer,loss = "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])


    if reduce_epoch:
        final_epochs = (epochs - 10) if epochs > 10 else epochs
    else:
        final_epochs = epochs

    if validation_split > 0:
        model_history = model.fit(train_ds, validation_split=0.04, epochs=final_epochs,
                                  verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
    else:

        model_history = model.fit(train_ds, validation_data=validation_ds, epochs=final_epochs, verbose=2,
                                  batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])


    # cope model parameters to another model
    model2 = model_Bidirectional_or_GRU_mask1(N_outputs, 3, [32, 64, 128, 256], N_times, N_feature, False, 0.5, False,
                                            False, mask, "linear")

    for il, ilayer in enumerate(model.layers):
        ilayer1 = model.layers[il]
        ilayer2 = model2.layers[il]
        name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic != '_'])
        name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic != '_'])
        if name_cls == name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
            model2.layers[il].set_weights(model.layers[il].get_weights())
    model2.save(dd_model)

    hist_df = pd.DataFrame(model_history.history)
    with open(ff_hist, mode='w') as f:
        hist_df.to_csv(f)


## training with validation data
# @tf.function
def trainings_val_class(train_ds, validation_ds, dd_model, ff_hist, num_epochs, mask=True):
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_accuracy_results = []

    optimizers = []
    for i in range(4):
        # optimizers.append(tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE / 10 ** i, momentum=0.9))
        optimizers.append(tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10 ** i))

    # VALIDATION_STEP = math.ceil(len(list(train_ds))/3)
    VALIDATION_STEP = math.ceil(len(list(train_ds)) / 2.1)
    # VALIDATION_STEP = math.ceil(len(list(train_ds))/4.1)
 
    totali = 0
    which_rate = 0
    lasti = 0
    optimizer = optimizers[which_rate]
    validation_accuracies = []
    max_change_rate = len(optimizers)

    model = model_Bidirectional_or_GRU_mask(N_outputs, 3, [32, 64, 128, 256], N_times, N_feature, False, 0.5, False,False, mask, "linear")
    # model = model_Bidirectional_or_GRU_attention_mask(N_outputs, 3, [32, 64, 128, 256], N_times, N_feature, False, 0.5, False,
    #                                         False, mask, "linear")

    reg = False

    # tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)
    for epoch in range(num_epochs):
        print ('training at the epoch %s' % (epoch + 1))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches
        for x, y in train_ds:

            loss_value, grads = grad(model, x, y, reg, -9999)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
 
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            totali = totali + 1

            if (totali % VALIDATION_STEP == 0):
                for xv, yv in validation_ds:
                    epoch_val_accuracy.update_state(yv, model(xv))

                validation_accuracies.append(epoch_val_accuracy.result())
                currenti = len(validation_accuracies)

                if which_rate < (max_change_rate - 1) and (
                        currenti - lasti) > PLAT_N and currenti > PLAT_N and is_validation_flat(validation_accuracies):
                    which_rate = which_rate + 1
                    lasti = currenti
                    optimizer = optimizers[which_rate]
                    logging.info('learning rate is reduced to %s at validation step %s at the epoch %s' % (0.01 / 10 ** (which_rate - 1), currenti, epoch+1))
        
        train_loss_results.append(epoch_loss_avg.result().numpy())
        val_accuracy_results.append(epoch_val_accuracy.result().numpy())

    model.save(dd_model)
    ls_acc = ['epoch,train_loss,val_acc']
    for i in range(len(train_loss_results)):
        ls_acc.append('%s,%s,%s' % (i+1,train_loss_results[i],val_accuracy_results[i]))
    with open(ff_hist, 'w') as _fo:
        _fo.write('\n'.join(ls_acc))
        
        
