import logging
import os
import xarray as xr
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.layers.convolutional import Conv1D
from tensorflow.keras import backend as K

from data_utils import data_utils

mli_mean = xr.open_dataset('norm/input_mean.nc')
mli_min = xr.open_dataset('norm/input_min.nc')
mli_max = xr.open_dataset('norm/input_max.nc')
mlo_scale = xr.open_dataset('norm/output_scale.nc')
grid_path = 'grid_info/ClimSim_low-res_grid-info.nc'
grid_info = xr.open_dataset(grid_path)

data = data_utils(grid_info=grid_info,
                  input_mean=mli_mean,
                  input_max=mli_max,
                  input_min=mli_min,
                  output_scale=mlo_scale)

data.set_to_v1_vars()

train_input = data.reshape_input_for_cnn(np.load('train/npy_data/train_input.npy'))
train_output = data.reshape_target_for_cnn(np.load('train/npy_data/train_target.npy'))

val_input = data.reshape_input_for_cnn(np.load('train/npy_data/val_input.npy'))
val_output = data.reshape_target_for_cnn(np.load('train/npy_data/val_target.npy'))


def mse_adjusted(y_true, y_pred):
    se = K.square(y_pred - y_true)
    return K.mean(se[:, :, 0:2]) * (120 / 128) + K.mean(se[:, :, 2:10]) * (8 / 128)


def mae_adjusted(y_true, y_pred):
    ae = K.abs(y_pred - y_true)
    return K.mean(ae[:, :, 0:2]) * (120 / 128) + K.mean(ae[:, :, 2:10]) * (8 / 128)

def continuous_ranked_probability_score(y_true, y_pred):
    """Continuous Ranked Probability Score.

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.
    We've closly followed the aproach of
    https://github.com/TheClimateCorporation/properscoring for
    for the actual implementation.

    Reference
    ---------
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
        Tensors of same shape and type.
    """
    score = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)), axis=-1)
    diff = tf.subtract(tf.expand_dims(y_pred, -1), tf.expand_dims(y_pred, -2))
    score = tf.add(score, tf.multiply(tf.constant(-0.5, dtype=diff.dtype),tf.reduce_mean(tf.abs(diff),axis=(-2, -1))))

    return tf.reduce_mean(score)

def createModel():
    """
            Create a ResNet-style 1D CNN. The data is of shape (batch, lev, vars)
            where lev is treated as the spatial dimension. The architecture
            consists of residual blocks with each two conv layers.
            """
    # Define output shapes
    in_shape = (60, 6)
    out_shape = (60, 10)
    output_length_lin = 2
    output_length_relu = out_shape[-1] - 2

    hp_depth = 12
    hp_channel_width = 406
    hp_kernel_width = 3
    hp_activation = "relu"
    hp_pre_out_activation = "elu"
    hp_norm = False
    hp_dropout = 0.175
    hp_optimizer = "Adam"
    hp_loss = "mean_absolute_error"

    channel_dims = [hp_channel_width] * hp_depth
    kernels = [hp_kernel_width] * hp_depth

    # Initialize special layers

    if hp_norm == "layer_norm":
        norm_layer = tf.keras.layers.LayerNormalization(axis=1)
    elif hp_norm == "batch_norm":
        norm_layer = tf.keras.layers.BatchNormalization()
    else:
        norm_layer = None

    if len(channel_dims) != len(kernels):
        print(
            f"[WARNING] Length of channel_dims and kernels does not match. Using 1st argument in kernels, {kernels[0]}, for every layer"
        )
        kernels = [kernels[0]] * len(channel_dims)

    # Initialize model architecture
    input_layer = keras.Input(shape=in_shape)
    x = input_layer  # Set aside input layer
    previous_block_activation = x  # Set aside residual
    for filters, kernel_size in zip(channel_dims, kernels):
        # First conv layer in block
        # 'same' applies zero padding.
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
        # todo: add se_block
        if norm_layer:
            x = norm_layer(x)
        x = keras.layers.Activation(hp_activation)(x)
        x = keras.layers.Dropout(hp_dropout)(x)

        # Second convolution layer
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
        if norm_layer:
            x = norm_layer(x)
        x = keras.layers.Activation(hp_activation)(x)
        x = keras.layers.Dropout(hp_dropout)(x)

        # Project residual
        residual = Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="same"
        )(previous_block_activation)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Output layers.
    # x = keras.layers.Dense(filters[-1], activation='gelu')(x) # Add another last layer.
    x = Conv1D(
        out_shape[-1],
        kernel_size=1,
        activation=hp_pre_out_activation,
        padding="same",
    )(x)
    # Assume that vertically resolved variables follow no particular range.
    output_lin = keras.layers.Dense(output_length_lin, activation="linear")(x)
    # Assume that all globally resolved variables are positive.
    output_relu = keras.layers.Dense(output_length_relu, activation="relu")(x)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name="cnn")

    # Optimizer
    # Set up cyclic learning rate
    INIT_LR = 1e-4
    MAX_LR = 1e-3
    steps_per_epoch = 10091520 // hp_depth

    # Set up optimizer
    if hp_optimizer == "Adam":
        my_optimizer = keras.optimizers.Adam()
    elif hp_optimizer == "SGD":
        my_optimizer = keras.optimizers.SGD()

    if hp_loss == "mse":
        loss = mse_adjusted
    elif hp_loss == "mean_absolute_error":
        loss = mae_adjusted
    elif hp_loss == "kl_divergence":
        loss = tf.keras.losses.KLDivergence()
    # compile
    model.compile(
        optimizer=my_optimizer,
        loss=loss,
        metrics=["mse", "mae", "accuracy", mse_adjusted, mae_adjusted, continuous_ranked_probability_score],
    )

    print(model.summary())

    return model

def train(model):
    path = "./models/" + modelName + "/"
    # callbacks
    # a. tensorboard
    tboard_callback = keras.callbacks.TensorBoard(log_dir=path + 'logs_tensorboard',
                                                  histogram_freq=1, )

    # b. checkpoint
    filepath_checkpoint = path + "model.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=filepath_checkpoint,
                                                          save_weights_only=False,
                                                          monitor='val_mse',
                                                          mode='min',
                                                          save_best_only=True)

    # c. csv logger
    filepath_csv = path + 'csv_logger.txt'
    csv_callback = keras.callbacks.CSVLogger(filepath_csv, separator=",", append=True)

    my_callbacks = [tboard_callback, checkpoint_callback, csv_callback]

    max_epochs = 15

    model.fit(
        train_input,
        train_output,
        epochs=max_epochs,
        validation_data=(val_input, val_output),
        verbose=1,
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
        callbacks=my_callbacks
    )

modelName = "cnn4"
model = createModel()
train(model)