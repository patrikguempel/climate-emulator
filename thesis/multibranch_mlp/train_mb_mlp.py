import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Loading Data...")

train_x = np.load("train/npy_data/train_input.npy")
train_y = np.load("train/npy_data/train_target.npy")
val_x = np.load("train/npy_data/val_input.npy")
val_y = np.load("train/npy_data/val_target.npy")

logging.info("Data loaded!")

def createModel():

    input_layer = Input(shape=(124,))

    # Shared Layer
    shared = layers.Dense(512, activation="relu")

    # Branches
    outputs = []
    for i in range(10):
        if i < 2:
            branch_start = layers.Dense(512, activation='relu')(shared)
            out_hid = layers.Dense(256, activation='relu')(branch_start)
            out_hid2 = layers.Dense(128, activation='elu')(out_hid)
            out_lin = layers.Dense(60, activation='linear')(out_hid2)
            outputs += [out_lin]
        else:
            branch_start = layers.Dense(128, activation='relu')(shared)
            out_hid = layers.Dense(16, activation='elu')(branch_start)
            out_lin = layers.Dense(1, activation='linear')(out_hid)
            outputs += [out_lin]

    output_layer = layers.Concatenate()(outputs)

    model = Model(input_layer, output_layer, name='MB_MLP')
    model.summary()

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse', 'mae', 'accuracy'])

    return model
def trainModel(model):
    logging.info("Model created!")


    path = "./models/" + model_name + "/"
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

    model.fit(train_x,
              train_y,
              epochs=30,
              verbose=1,
              validation_data=(val_x, val_y),
              callbacks=my_callbacks,
              use_multiprocessing=True,
              workers=4,)

model_name = "mb_mlp3"
model = createModel()
trainModel(model)