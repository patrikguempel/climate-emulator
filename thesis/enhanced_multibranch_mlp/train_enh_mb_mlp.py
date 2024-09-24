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

# Load Data
train_input = np.load('train/npy_data/train_input.npy')
train_output = np.load('train/npy_data/train_target.npy')
val_input = np.load('train/npy_data/val_input.npy')
val_output = np.load('train/npy_data/val_target.npy')
logging.info("Data loaded!")


def create_model():
    # Input Layer
    input_layer = Input(shape=(124,), name='input')

    # Shared Layer
    shared_layer = Dense(512, activation="relu")(input_layer)

    # 10 Upper Branches (-> 12)
    upper_branches_outputs = []
    for i in range(10):
        if i < 2:
            hid_1 = Dense(512, activation="relu")(shared_layer)
            hid_2 = Dense(256, activation="relu")(hid_1)

            # split
            upper_atmosphere = Lambda(lambda x: x[:, :128])(hid_2)
            lower_atmosphere = Lambda(lambda x: x[:, 128:])(hid_2)

            upper_branches_outputs += [upper_atmosphere]
            upper_branches_outputs += [lower_atmosphere]
        else:
            hid_1 = Dense(128, activation="relu")(shared_layer)

            upper_branches_outputs += [hid_1]

    # merge to one tensor and normalize
    ca_tensor = tf.stack(upper_branches_outputs, axis=1)
    ca_tensor_normalized = LayerNormalization()(ca_tensor)

    # attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=128)(ca_tensor_normalized, ca_tensor_normalized)

    # skip connection
    skip_connection = Add()([ca_tensor_normalized, LayerNormalization()(attention_output)])
    skip_connection = LayerNormalization()(skip_connection)

    # 12 Lower Branches
    lower_branch_outputs = []
    for i in range(12):
        branch = Lambda(lambda x: x[:, i, :])(skip_connection)
        if i < 4:
            hid_3 = Dense(64, activation="elu")(branch)
            out   = Dense(30, activation="linear")(hid_3)
        else:
            hid_3 = Dense(16, activation="elu")(branch)
            out   = Dense(1, activation="linear")(hid_3)
        lower_branch_outputs += [out]

    output_layer = Concatenate()(lower_branch_outputs)

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


model_name = "mlp_ca8-3"
model = create_model()
trainModel(model)
