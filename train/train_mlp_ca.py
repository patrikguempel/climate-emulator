import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LayerNormalization, Input, Concatenate, MultiHeadAttention, Add, LeakyReLU

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
    def cross_attention(query, key, value):
        attn = MultiHeadAttention(num_heads=4, key_dim=32)  # Anzahl der Köpfe und Dimensionen anpassbar
        return attn(query, key, value)

    input_layer = Input(shape=(124,), name='input')
    norm_layer = LayerNormalization()(input_layer)

    branch_1 = tf.expand_dims(norm_layer[:, :60], axis=-1)
    branch_2 = tf.expand_dims(norm_layer[:, 60:120], axis=-1)
    branch_3 = tf.expand_dims(tf.expand_dims(norm_layer[:, 120], axis=-1), axis=-1)
    branch_4 = tf.expand_dims(tf.expand_dims(norm_layer[:, 121], axis=-1), axis=-1)
    branch_5 = tf.expand_dims(tf.expand_dims(norm_layer[:, 122], axis=-1), axis=-1)
    branch_6 = tf.expand_dims(tf.expand_dims(norm_layer[:, 123], axis=-1), axis=-1)

    # Cross-Attention
    attn_1 = cross_attention(branch_1,
                             tf.concat([branch_2, branch_3, branch_4, branch_5, branch_6], axis=1),
                             tf.concat([branch_2, branch_3, branch_4, branch_5, branch_6], axis=1))

    attn_2 = cross_attention(branch_2,
                             tf.concat([branch_1, branch_3, branch_4, branch_5, branch_6], axis=1),
                             tf.concat([branch_1, branch_3, branch_4, branch_5, branch_6], axis=1))

    attn_3 = cross_attention(branch_3,
                             tf.concat([branch_1, branch_2, branch_4, branch_5, branch_6], axis=1),
                             tf.concat([branch_1, branch_2, branch_4, branch_5, branch_6], axis=1))

    attn_4 = cross_attention(branch_4,
                             tf.concat([branch_1, branch_2, branch_3, branch_5, branch_6], axis=1),
                             tf.concat([branch_1, branch_2, branch_3, branch_5, branch_6], axis=1))

    attn_5 = cross_attention(branch_5,
                             tf.concat([branch_1, branch_2, branch_3, branch_4, branch_6], axis=1),
                             tf.concat([branch_1, branch_2, branch_3, branch_4, branch_6], axis=1))

    attn_6 = cross_attention(branch_6,
                             tf.concat([branch_1, branch_2, branch_3, branch_4, branch_5], axis=1),
                             tf.concat([branch_1, branch_2, branch_3, branch_4, branch_5], axis=1))

    #merge
    merged_branches = Concatenate(axis=1)([attn_1, attn_2, attn_3, attn_4, attn_5, attn_6])

    output_ca = Add()([input_layer, merged_branches])    #or add to norm layer?

    norm_layer2 = LayerNormalization()(output_ca)

    hidden_0 = Dense(768, activation=LeakyReLU(alpha=0.15))(norm_layer2)
    hidden_1 = Dense(640, activation=LeakyReLU(alpha=0.15))(hidden_0)
    hidden_2 = Dense(512, activation=LeakyReLU(alpha=0.15))(hidden_1)
    hidden_3 = Dense(640, activation=LeakyReLU(alpha=0.15))(hidden_2)
    hidden_4 = Dense(640, activation=LeakyReLU(alpha=0.15))(hidden_3)
    hidden_5 = Dense(128, activation=LeakyReLU(alpha=0.15))(hidden_4)
    out1 = Dense(120, activation='linear')(hidden_5)
    out2 = Dense(8, activation='relu')(hidden_5)
    output_layer_mlp = Concatenate()([out1, out2])

    output_layer = Add()([output_ca, output_layer_mlp])

    model = keras.Model(input_layer, output_layer, name='Emulator_CA')
    model.summary()

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

model_name = "mlp_ca1"

model = create_model()
trainModel(model)

