import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Concatenate, MultiHeadAttention, Add, LeakyReLU, LayerNormalization, Flatten

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
        attn = MultiHeadAttention(num_heads=4, key_dim=32)  # num_heads and key_dim are adjustable
        return attn(query, key, value)


    # Input Layer
    input_layer = Input(shape=(124,), name='input')

    # Branch for each feature
    branch_1 = tf.expand_dims(input_layer[:, :60], axis=-1)
    branch_2 = tf.expand_dims(input_layer[:, 60:120], axis=-1)
    branch_3 = tf.expand_dims(tf.expand_dims(input_layer[:, 120], axis=-1), axis=-1)
    branch_4 = tf.expand_dims(tf.expand_dims(input_layer[:, 121], axis=-1), axis=-1)
    branch_5 = tf.expand_dims(tf.expand_dims(input_layer[:, 122], axis=-1), axis=-1)
    branch_6 = tf.expand_dims(tf.expand_dims(input_layer[:, 123], axis=-1), axis=-1)
    branches_raw = [branch_1, branch_2, branch_3, branch_4, branch_5, branch_6]

    # Linear layers
    branches_lin = [Dense(branch.shape[-1], activation='linear')(branch) for branch in branches_raw]

    # Normalization
    branches = [LayerNormalization(axis=-1)(branch) for branch in branches_lin]

    # Cross-Attention
    attention = []
    for i in range(len(branches)):
        branch_complement = tf.concat([branches[x] for x in range(len(branches)) if x != i], axis=1)
        attention += [cross_attention(branches[i], branch_complement, branch_complement)]



    # Skip-Connection
    intermediate_results = [Add()([branches_lin[i], attention[i]]) for i in range(len(branches))]

    # Normalization
    branches = [LayerNormalization(axis=-1)(branch) for branch in intermediate_results]

    # Flatten and prepare for MLP
    branch_outputs = []
    for i in range(len(branches)):
        if i < 2:
            pre_0 = Flatten()(branches[i])
            pre_1 = Dense(60, activation='relu')(pre_0)
            branch_outputs += [pre_1]
        else:
            pre_0 = Flatten()(branches[i])
            pre_1 = Dense(10, activation='relu')(pre_0)
            branch_outputs += [pre_1]


    concatenation = Concatenate(axis=-1)(branch_outputs)

    # Employ MLP model (which is a scaled-down adjusted version of the multi-branch mlp)
    hidden_0 = Dense(768, activation=LeakyReLU(alpha=0.15))(concatenation)
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

    # Skip connection not possible, because dimensions have changed

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
              workers=4,)

model_name = "mlp_ca1"

model = create_model()
trainModel(model)

