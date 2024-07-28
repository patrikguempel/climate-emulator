import xarray as xr
import tensorflow as tf
from tensorflow import keras
import glob
import random
from keras.layers.convolutional import Conv1D
from keras import backend as K


mli_mean = xr.open_dataset('./norm/input_mean.nc')
mli_min = xr.open_dataset('./norm/input_min.nc')
mli_max = xr.open_dataset('./norm/input_max.nc')
mlo_scale = xr.open_dataset('./norm/output_scale.nc')

# in/out variable lists
vars_mli = ['state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC',
            'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']


def loadNCdir(filelist: list):
    def gen():
        for file in filelist:
            # read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            ds = ds[vars_mli]

            # read mlo
            dso = xr.open_dataset(file.replace('.mli.', '.mlo.'), engine='netcdf4')

            # make mlo variales: ptend_t and ptend_q0001
            dso['ptend_t'] = (dso['state_t'] - ds['state_t']) / 1200  # T tendency [K/s]
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001']) / 1200  # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]

            # normalizatoin, scaling
            ds = (ds - mli_mean) / (mli_max - mli_min)
            dso = dso * mlo_scale

            # stack
            # ds = ds.stack({'batch':{'sample','ncol'}})
            ds = ds.stack({'batch': {'ncol'}})
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            # dso = dso.stack({'batch':{'sample','ncol'}})
            dso = dso.stack({'batch': {'ncol'}})
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

            yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(gen,
                                        output_signature=(
                                            tf.TensorSpec(shape=(60, 6), dtype=tf.float32),
                                            tf.TensorSpec(shape=(60, 10), dtype=tf.float32)))


def main():
    modelName = "mlp1"

    f_mli, f_mli_val = getDataPaths(stride_sample=19)     # (subsampling is done here by "stride_sample")
    model: keras.Model = createModel()
    train(f_mli, f_mli_val, model, modelName)


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


def getDataPaths(stride_sample=19):
    #original stride sample was 37
    #because here I only use 5 years of simulation (instead of 10), i wanna use about two times the data then
    f_mli1 = glob.glob('../../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.000[1234]-*-*-*.nc')
    f_mli2 = glob.glob('../../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0005-01-*-*.nc')
    f_mli = sorted([*f_mli1, *f_mli2])
    random.shuffle(f_mli)  # to reduce IO bottleneck
    f_mli = f_mli[::stride_sample]

    # validation dataset for HPO
    f_mli1 = glob.glob('../../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0005-0[23456789]-*-*.nc')
    f_mli2 = glob.glob('../../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0005-1[012]-*-*.nc')
    f_mli3 = glob.glob('../../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0006-01-*-*.nc')
    f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
    random.shuffle(f_mli_val)
    f_mli_val = f_mli_val[::stride_sample]

    return f_mli, f_mli_val


def train(f_mli, f_mli_val, model: keras.Model, modelName: str, n_epochs: int = 30, shuffle_buffer: int = 12 * 384,
          batch_size=96):  # ncol = 384      384/4 = 96

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

    tds = loadNCdir(f_mli) \
            .unbatch() \
            .repeat(max_epochs) \
            .shuffle(shuffle_buffer) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(tf.data.AUTOTUNE)

    tds_val = loadNCdir(f_mli_val) \
            .unbatch() \
            .shuffle(shuffle_buffer) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

    print("Data loaded!")

    model.fit(
        tds,
        epochs=max_epochs,
        steps_per_epoch=10091520 // batch_size,
        validation_data=tds_val,
        verbose=1,
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
        callbacks=my_callbacks
    )

main()