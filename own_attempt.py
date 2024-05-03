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

def main():
    # in/out variable lists
    vars_mli = ['state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
    vars_mlo = ['ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC',
                'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

    # normalization/scaling factors
    mli_mean = xr.open_dataset('../../norm_factors/mli_mean.nc', engine='netcdf4')
    mli_min = xr.open_dataset('../../norm_factors/mli_min.nc', engine='netcdf4')
    mli_max = xr.open_dataset('../../norm_factors/mli_max.nc', engine='netcdf4')
    mlo_scale = xr.open_dataset('../../norm_factors/mlo_scale.nc', engine='netcdf4')

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