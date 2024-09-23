import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input, Concatenate, MultiHeadAttention, Add, Lambda, \
    LayerNormalization

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

    # Separate into 6 branches
    branch_inputs = []
    branch_inputs += [input_layer[:, :60]]
    branch_inputs += [input_layer[:, 60:120]]
    branch_inputs += [input_layer[:, 120:121]]
    branch_inputs += [input_layer[:, 121:122]]
    branch_inputs += [input_layer[:, 122:123]]
    branch_inputs += [input_layer[:, 123:124]]

    # Map to 6 same size linear layers
    branches = [Dense(64, activation='linear')(branch_inputs[i]) for i in range(6)]

    # Expand the branches, give them a channel dimension
    expanded_branches = [tf.expand_dims(branch, axis=1) for branch in branches]

    # Concatenate them along axis 1 to get to (batch, 6, 64) and normalize
    concat_branches = tf.concat(expanded_branches, axis=1)
    concat_branches_normalized = LayerNormalization()(concat_branches)

    # Cross attention
    attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(concat_branches_normalized, concat_branches_normalized)

    # skip connection: add two normalized tensors
    checkpoint = Add()([concat_branches_normalized, LayerNormalization()(attention_output)])

    # normalize mlp input
    normalized_mlp_input = LayerNormalization()(checkpoint)

    # mlp on each individual branch
    branch_outputs = []
    for i in range(6):
        branch = Lambda(lambda x: x[:, i, :])(normalized_mlp_input)

        hid_1 = Dense(512, activation='relu')(branch)
        hid_2 = Dense(256, activation='relu')(hid_1)
        hid_3 = Dense(256, activation='relu')(hid_2)
        hid_4 = Dense(128, activation='elu')(hid_3)
        out_lin = Dense(128, activation='linear')(hid_4)
        branch_outputs += [out_lin]

    merged_output = Concatenate(axis=1)([tf.expand_dims(branch, axis=1) for branch in branch_outputs])

    # Reduce by taking the average
    reduced_output = GlobalAveragePooling1D()(merged_output)

    # Define model
    model = keras.Model(input_layer, reduced_output, name='Emulator_CA')
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


model_name = "mlp_ca7-2"
model = create_model()
trainModel(model)
