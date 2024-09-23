import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Concatenate, MultiHeadAttention, Add, LeakyReLU, \
    LayerNormalization, Flatten

import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

train_input = np.load('train/npy_data/train_input.npy')
train_output = np.load('train/npy_data/train_target.npy')
val_input = np.load('train/npy_data/val_input.npy')
val_output = np.load('train/npy_data/val_target.npy')
logging.info("Data loaded!")


def create_model():
    # Input Layer
    input_layer = Input(shape=(124,), name='input')

    shared = Dense(512, activation='relu')(input_layer)

    intermediate_outputs = []
    for i in range(10):
        if i < 2:
            branch_start = Dense(512, activation='relu')(shared)
            hid_1 = Dense(256, activation='relu')(branch_start)
            out_hid2 = Dense(128, activation='elu')(hid_1)
            out_lin = Dense(60, activation='linear')(out_hid2)
        else:
            branch_start = Dense(128, activation='relu')(shared)
            out_hid2 = Dense(16, activation='elu')(branch_start)
            out_lin = Dense(1, activation='linear')(out_hid2)

        intermediate_outputs += [out_lin]

    attention_outputs = []
    branch_sizes = [60, 60, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(10):
        other_branches = Concatenate()([intermediate_outputs[j] for j in range(10) if j != i])

        query = tf.expand_dims(intermediate_outputs[i], axis=1)
        key_value = tf.expand_dims(other_branches, axis=1)
        cross_attention = MultiHeadAttention(num_heads=4, key_dim=32)(query, key_value, key_value)

        cross_attention = tf.squeeze(cross_attention, axis=1)

        skip_connection = Add()([cross_attention, intermediate_outputs[i]])

        final_layer = Dense(branch_sizes[i], activation='elu')(skip_connection)

        attention_outputs += [final_layer]

    output_layer = Concatenate()(attention_outputs)

    # Define model
    model = keras.Model(input_layer, output_layer, name='Emulator_CA')
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
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
              workers=4, )


model_name = "mlp_ca3_pluselu"

model = create_model()
trainModel(model)
