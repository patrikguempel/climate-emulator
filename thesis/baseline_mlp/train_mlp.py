import logging
import os

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Concatenate

# Load Data
train_input = np.load('train/npy_data/train_input.npy')
train_output = np.load('train/npy_data/train_target.npy')
val_input = np.load('train/npy_data/val_input.npy')
val_output = np.load('train/npy_data/val_target.npy')

def createModel():
    # construct
    input_layer = Input(shape=(124,), name='input')
    hidden_0 = Dense(768, activation=LeakyReLU(alpha=0.15))(input_layer)
    hidden_1 = Dense(640, activation=LeakyReLU(alpha=0.15))(hidden_0)
    hidden_2 = Dense(512, activation=LeakyReLU(alpha=0.15))(hidden_1)
    hidden_3 = Dense(640, activation=LeakyReLU(alpha=0.15))(hidden_2)
    hidden_4 = Dense(640, activation=LeakyReLU(alpha=0.15))(hidden_3)
    hidden_5 = Dense(128, activation=LeakyReLU(alpha=0.15))(hidden_4)
    out1     = Dense(120, activation='linear')(hidden_5)
    out2     = Dense(8  , activation='relu')(hidden_5)
    output_layer = Concatenate()([out1, out2])

    model = keras.Model(input_layer, output_layer, name='Emulator')
    model.summary()

    # compile
    model.compile(optimizer=keras.optimizers.Adam(),
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

model_name = "mlp_optimized"

model = createModel()
trainModel(model)