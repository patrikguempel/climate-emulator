import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

model_name = "mb_mlp_connected"

logging.info("Loading Data...")

train_x = np.load("train/npy_data/train_input.npy")
train_y = np.load("train/npy_data/train_target.npy")
val_x = np.load("train/npy_data/val_input.npy")
val_y = np.load("train/npy_data/val_target.npy")

logging.info("Data loaded!")

def createModel():

    input_layer = Input(shape=(124,))
    start = layers.Dense(1792, activation='relu')(input_layer)
    hid1 = layers.Dense(320, activation='relu')(start)
    hid2 = layers.Dense(128, activation='elu')(start)
    hid = layers.Concatenate()([hid1, hid2])
    out_hid = layers.Dense(256, activation='elu')(hid)
    out_1 = layers.Dense(120, activation='linear')(out_hid)
    out_2 = layers.Dense(8, activation='linear')(hid)
    output_layer = layers.Concatenate()([out_1, out_2])

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

model = createModel()
#trainModel(model)