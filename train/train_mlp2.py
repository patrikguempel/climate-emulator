import logging
import os

import numpy as np
from tensorflow import keras
import tensorflow as tf

train_input = np.load('train/npy_data/train_input.npy')
train_output = np.load('train/npy_data/train_target.npy')

# validation data
val_input = np.load('train/npy_data/val_input.npy')
val_output = np.load('train/npy_data/val_target.npy')

def createModel():
    input_length = 2 * 60 + 4
    output_length_lin = 2 * 60
    output_length_relu = 8
    output_length = output_length_lin + output_length_relu
    n_nodes = 512

    # construct a model
    input_layer = keras.layers.Input(shape=(input_length,), name='input')
    hidden_0 = keras.layers.Dense(n_nodes, activation='relu')(input_layer)
    hidden_1 = keras.layers.Dense(n_nodes, activation='relu')(hidden_0)
    output_pre = keras.layers.Dense(output_length, activation='elu')(hidden_1)
    output_lin = keras.layers.Dense(output_length_lin, activation='linear')(output_pre)
    output_relu = keras.layers.Dense(output_length_relu, activation='relu')(output_pre)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name='Emulator')
    model.summary()

    # compile
    model.compile(optimizer=keras.optimizers.Adam(),  # optimizer=keras.optimizers.Adam(learning_rate=clr),
                  loss='mse',
                  metrics=['mse', 'mae', 'accuracy'])

    return model

def trainModel(model):
    logging.info("Model created!")


    path = "./models/" + model_name + "/"
    os.makedirs(path, exist_ok=True)
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path + 'logs_tensorboard',
                                                  histogram_freq=1, )

    # b. checkpoint
    filepath_checkpoint = path + "model.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_checkpoint,
                                                          save_weights_only=False,
                                                          monitor='val_mse',
                                                          mode='min',
                                                          save_best_only=True)

    # c. csv logger
    filepath_csv = path + 'csv_logger.txt'
    csv_callback = tf.keras.callbacks.CSVLogger(filepath_csv, separator=",", append=True)

    my_callbacks = [tboard_callback, checkpoint_callback, csv_callback]

    model.fit(train_input,
              train_output,
              epochs=30,
              verbose=1,
              validation_data=(val_input, val_output),
              callbacks=my_callbacks,
              use_multiprocessing=True,
              workers=4,)

model_name = "mlp2"

model = createModel()
trainModel(model)