import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import callbacks
from tensorflow.keras.optimizers import Adam
import os
import argparse
import glob
import random
import sys

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

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((None, 124), (None, 128))
    )


def main():
    # train dataset for HPO
    # (subsampling id done here by "stride_sample")

    remote = sys.argv[1] == "remote"

    f_mli, f_mli_val = getDataPaths(remote=False)
    model: keras.Model = createModel()
    train(f_mli, f_mli_val, model)


def createModel():
    input_length = 2 * 60 + 4
    output_length_lin = 2 * 60
    output_length_relu = 8
    output_length = output_length_lin + output_length_relu
    n_nodes = 512

    # construct a model
    input_layer = keras.layers.Input(shape=(input_length,), name='input')
    hidden_0 = keras.layers.Dense(n_nodes, activation='relu')(input_layer)
    hidden_1 = keras.layers.Dense(n_nodes, activation='relu')(hidden_0)
    output_pre = keras.layers.Dense(output_length, activation='elu')(hidden_1)
    output_lin = keras.layers.Dense(output_length_lin, activation='linear')(output_pre)
    output_relu = keras.layers.Dense(output_length_relu, activation='relu')(output_pre)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name='Emulator')
    model.summary()

    # compile
    model.compile(optimizer=keras.optimizers.Adam(),  # optimizer=keras.optimizers.Adam(learning_rate=clr),
                  loss='mse',
                  metrics=['mse', 'mae', 'accuracy'])

    return model


def getDataPaths(remote: bool, stride_sample: int = 37):
    if remote:
        f_mli1 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.000[1234567]-*-*-*.nc')
        f_mli2 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0008-01-*-*.nc')
        f_mli = sorted([*f_mli1, *f_mli2])
        random.shuffle(f_mli)  # to reduce IO bottleneck
        f_mli = f_mli[::stride_sample]

        # validation dataset for HPO
        f_mli1 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0008-0[23456789]-*-*.nc')
        f_mli2 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0008-1[012]-*-*.nc')
        f_mli3 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0009-01-*-*.nc')
        f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
        random.shuffle(f_mli_val)
        f_mli_val = f_mli_val[::stride_sample]
    else:
        f_mli = ["./sampledata/sample.nc"]
        f_mli_val = ["./sampledata/sample2.nc"]

    return f_mli, f_mli_val


def train(f_mli, f_mli_val, model: keras.Model, n_epochs: int = 30, shuffle_buffer: int = 12 * 384, batch_size=96):  # ncol = 384      384/4 = 96
    for n in range(n_epochs):
        random.shuffle(f_mli)
        tds = loadNCdir(f_mli)  # global shuffle by file names
        tds = tds.unbatch()
        # local shuffle by elements    tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
        tds = tds.batch(batch_size)
        tds = tds.prefetch(buffer_size=int(shuffle_buffer / 384))  # in realtion to the batch size

        random.shuffle(f_mli_val)
        tds_val = loadNCdir(f_mli_val)
        tds_val = tds_val.unbatch()
        tds_val = tds_val.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
        tds_val = tds_val.batch(batch_size)
        tds_val = tds_val.prefetch(buffer_size=int(shuffle_buffer / 384))

        print(f'Epoch: {n + 1}')
        model.fit(tds,
                  validation_data=tds_val,
              #    callbacks=my_callbacks
                  )

main()