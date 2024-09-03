from data_utils import data_utils
import xarray as xr

mli_mean = xr.open_dataset('norm/input_mean.nc')
mli_min = xr.open_dataset('norm/input_min.nc')
mli_max = xr.open_dataset('norm/input_max.nc')
mlo_scale = xr.open_dataset('norm/output_scale.nc')
grid_info = xr.open_dataset('grid_info/ClimSim_low-res_grid-info.nc')


data = data_utils(grid_info=grid_info,
                  input_mean=mli_mean,
                  input_max=mli_max,
                  input_min=mli_min,
                  output_scale=mlo_scale)

data.set_to_v2_vars()
data.data_path = '../climsim-dataset/ClimSim_low-res/train/'

print("Creating training data...")

data.set_regexps(data_split="train",
                 regexps=["E3SM-MMF.mli.000[1234]-*-*-*.nc",
                          "E3SM-MMF.mli.0005-01-*-*.nc"])
data.set_stride_sample(data_split="train", stride_sample=19)
data.set_filelist(data_split="train")
data.save_as_npy(data_split="train", save_path="train/npy_data_v2/")

print("Creating validation data...")

data.set_regexps(data_split="val",
                 regexps=["E3SM-MMF.mli.0005-0[23456789]-*-*.nc",
                          "E3SM-MMF.mli.0005-1[012]-*-*.nc",
                          "E3SM-MMF.mli.0006-01-*-*.nc"])
data.set_stride_sample(data_split="val", stride_sample=19)
data.set_filelist(data_split="val")
data.save_as_npy(data_split="val", save_path="train/npy_data_v2/")

print("Creating test data...")

data.set_regexps(data_split="val",
                 regexps=["E3SM-MMF.mli.0006-0[23456789]-*-*.nc",
                          "E3SM-MMF.mli.0006-1[012]-*-*.nc",
                          "E3SM-MMF.mli.0007-01-*-*.nc"])
data.set_stride_sample(data_split="val", stride_sample=19)
data.set_filelist(data_split="val")
data.save_as_npy(data_split="val", save_path="train/npy_data_v2/")

print("Done!")