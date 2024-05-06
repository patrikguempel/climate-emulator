import xarray
import tensorflow as tf
import tf.data.Dataset
from tensorflow.python import keras


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


dataset: Dataset = load_nc_dir_with_generator(["./sampledata/sample.nc"])
