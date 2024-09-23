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
    # Branch sizes
    branch_sizes = [60, 60, 1, 1, 1, 1, 1, 1, 1, 1]

    # Input Layer
    input_layer = Input(shape=(124,), name='input')

    # Branch Starts: Linear layer
    branch_starts = []
    for i in range(10):
        branch_start = Dense(branch_sizes[i], activation='linear')(input_layer)
        branch_starts += [branch_start]

    # Cross Attention
    attention_outputs = []
    for i in range(10):
        other_branches = Concatenate()([branch_starts[j] for j in range(10) if j != i])

        query = tf.expand_dims(branch_starts[i], axis=1)
        key_value = tf.expand_dims(other_branches, axis=1)

        attention_layer = MultiHeadAttention(num_heads=4, key_dim=32)(query, key_value, key_value)

        attention_layer = tf.squeeze(attention_layer, axis=1)
        skip_connection = Add()([attention_layer, branch_starts[i]])

        attention_outputs += [skip_connection]

    # MLP branches
    outputs = []
    for i in range(10):
        if i < 2:
            hid_1 = Dense(512, activation='relu')(attention_outputs[i])
            hid_2 = Dense(256, activation='relu')(hid_1)
            out_hid2 = Dense(128, activation='elu')(hid_2)
            out_lin = Dense(60, activation='linear')(out_hid2)
        else:
            hid_1 = Dense(128, activation='relu')(attention_outputs[i])
            out_hid2 = Dense(16, activation='elu')(hid_1)
            out_lin = Dense(1, activation='linear')(out_hid2)

        outputs += [out_lin]

    output_layer = Add()([Concatenate()(outputs), Concatenate()(attention_outputs)])

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


model_name = "mlp_ca4_big"

model = create_model()
trainModel(model)
