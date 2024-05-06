import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras import models
from keras import layers
from keras import callbacks
import os
import argparse
import glob
import random

mli_mean = xr.open_dataset('./norm_factors/mli_mean.nc')
mli_min = xr.open_dataset('./norm_factors/mli_min.nc')
mli_max = xr.open_dataset('./norm_factors/mli_max.nc')
mlo_scale = xr.open_dataset('./norm_factors/mlo_scale.nc')

def load_nc_dir_with_generator(filelist: list):
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
    # in/out variable lists
    vars_mli = ['state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
    vars_mlo = ['ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC',
                'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

    # normalization/scaling factors
    #mli_mean = xr.open_dataset('../../norm_factors/mli_mean.nc', engine='netcdf4')
    #mli_min = xr.open_dataset('../../norm_factors/mli_min.nc', engine='netcdf4')
    #mli_max = xr.open_dataset('../../norm_factors/mli_max.nc', engine='netcdf4')
    #mlo_scale = xr.open_dataset('../../norm_factors/mlo_scale.nc', engine='netcdf4')

    # train dataset for HPO
    # (subsampling id done here by "stride_sample")
    stride_sample = 37  # about ~20% assuming we will use 1/7 subsampled dataset for full training.
    f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[1234567]-*-*-*.nc')
    f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-01-*-*.nc')
    f_mli = sorted([*f_mli1, *f_mli2])
    random.shuffle(f_mli)  # to reduce IO bottleneck
    f_mli = f_mli[::stride_sample]

    # validation dataset for HPO
    f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-0[23456789]-*-*.nc')
    f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-1[012]-*-*.nc')
    f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0009-01-*-*.nc')
    f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
    random.shuffle(f_mli_val)
    f_mli_val = f_mli_val[::stride_sample]