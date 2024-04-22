import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import random

import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    for kgpu in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[kgpu], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# in/out variable lists
vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

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

shuffle_buffer=384*12

# for training

# # First 5 days of each month for the first 6 years
# f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[123456]-*-0[12345]-*.nc')
# f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-01-0[12345]-*.nc')
# f_mli = [*f_mli1, *f_mli2]

# every 10th sample
f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[123456]-*-*-*.nc')
f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-01-*-*.nc')
f_mli = sorted([*f_mli1, *f_mli2])
random.shuffle(f_mli)
f_mli = f_mli[::10]

# # debugging
# f_mli = f_mli[0:72*5]

random.shuffle(f_mli)
print(f'[TRAIN] Total # of input files: {len(f_mli)}')
print(f'[TRAIN] Total # of columns (nfiles * ncols): {len(f_mli)*384}')
tds = load_nc_dir_with_generator(f_mli)
tds = tds.unbatch()
tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
tds = tds.prefetch(buffer_size=4) # in realtion to the batch size

# for validation

# # First 5 days of each month for the following 2 years
# f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-0[23456789]-0[12345]-*.nc')
# f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-1[012]-0[12345]-*.nc')
# f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[89]-*-0[12345]-*.nc')
# f_mli_val = [*f_mli1, *f_mli2, *f_mli3]

# every 10th sample
f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-0[23456789]-0[12345]-*.nc')
f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0007-1[012]-0[12345]-*.nc')
f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[89]-*-0[12345]-*.nc')
f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
f_mli_val = f_mli_val[::10]

# # debugging
# f_mli_val = f_mli_val[0:72*5]

random.shuffle(f_mli_val)
print(f'[VAL] Total # of input files: {len(f_mli_val)}')
print(f'[VAL] Total # of columns (nfiles * ncols): {len(f_mli_val)*384}')
tds_val = load_nc_dir_with_generator(f_mli_val)
tds_val = tds_val.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
tds_val = tds_val.prefetch(buffer_size=4) # in realtion to the batch size

#list(tds)
# for count_batch in tds.repeat().batch(10).take(1):
#     print(count_batch[0].numpy())
#count_batch[0].shape

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():

# model params
input_length = 2*60 + 4
output_length_lin  = 2*60
output_length_relu = 8
output_length = output_length_lin + output_length_relu
n_nodes = 512

# constrcut a model
input_layer    = keras.layers.Input(shape=(input_length,), name='input')
hidden_0       = keras.layers.Dense(n_nodes, activation='relu')(input_layer)
hidden_1       = keras.layers.Dense(n_nodes, activation='relu')(hidden_0)
output_pre     = keras.layers.Dense(output_length, activation='elu')(hidden_1)
output_lin     = keras.layers.Dense(output_length_lin,activation='linear')(output_pre)
output_relu    = keras.layers.Dense(output_length_relu,activation='relu')(output_pre)
output_layer   = keras.layers.Concatenate()([output_lin, output_relu])

model = keras.Model(input_layer, output_layer, name='Emulator')
model.summary()

# compile
model.compile(optimizer=keras.optimizers.Adam(), #optimizer=keras.optimizers.Adam(learning_rate=clr),
              loss='mse',
              metrics=['mse','mae','accuracy'])

# callbacks
# a. tensorboard
tboard_callback = keras.callbacks.TensorBoard(log_dir = './logs_tensorboard',
                                              histogram_freq = 1,)

# b. checkpoint
filepath_checkpoint = 'saved_model/best_model_proto.h5'
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=filepath_checkpoint,
                                                            save_weights_only=False,
                                                            monitor='val_mse',
                                                            mode='min',
                                                            save_best_only=True)

# c. csv logger
filepath_csv = 'csv_logger.txt'
csv_callback = keras.callbacks.CSVLogger(filepath_csv, separator=",", append=True)

my_callbacks= [tboard_callback, checkpoint_callback, csv_callback]

# !mkdir logs_tensorboard
# !mkdir saved_model

# Manually shuffling the order of input files.
# "tds = tds.shuffle(buffer_size=<global>, reshuffle_each_iteration=True)" is possible,
# however, it is slow.
# So employing global shuffle (by file names) + local shuffle (using .shuffle).

N_EPOCHS = 30
shuffle_buffer = 12 * 384  # ncol=384
batch_size = 96  # 384/4

n = 0
while n < N_EPOCHS:
    random.shuffle(f_mli)
    tds = load_nc_dir_with_generator(f_mli)  # global shuffle by file names
    tds = tds.unbatch()
    # local shuffle by elements    tds = tds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=int(shuffle_buffer / 384))  # in realtion to the batch size

    random.shuffle(f_mli_val)
    tds_val = load_nc_dir_with_generator(f_mli_val)
    tds_val = tds_val.unbatch()
    tds_val = tds_val.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    tds_val = tds_val.batch(batch_size)
    tds_val = tds_val.prefetch(buffer_size=int(shuffle_buffer / 384))

    print(f'Epoch: {n + 1}')
    model.fit(tds,
              validation_data=tds_val,
              callbacks=my_callbacks)

    n += 1