import glob
import xarray as xr

mli_mean = xr.open_dataset('./norm/input_mean.nc')
mli_min = xr.open_dataset('./norm/input_min.nc')
mli_max = xr.open_dataset('./norm/input_max.nc')
mlo_scale = xr.open_dataset('./norm/output_scale.nc')

# in/out variable lists
vars_mli = ['state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC',
            'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

def pad_and_stack_layers_and_vars_1d(ds, dso):
    """
    Pads and stack all variables into (batch, n_vertical_levels, n_variables),
    e.g., input: (batch, 60, 6) and output: (batch, 60, 10)
    Args:
        ds xarray.Dataset(lev, ncol) with vars_mli of shapes (lev, ncol) and (ncol)
        dso xarray.Dataset(lev, ncol) with vars_mlo of shapes (lev, ncol) and (ncol)
    Returns:
        arr xarray.DataArray(batch, lev, variable)
        arro xarray.DataArray(batch, lev, variable)
    """
    ds = ds.stack({"batch": {"ncol"}})
    (ds,) = xr.broadcast(ds)  # repeat global variables across levels
    arr = ds.to_array("mlvar", name="mli")
    arr = arr.transpose("batch", "lev", "mlvar")

    dso = dso.stack({"batch": {"ncol"}})
    (dso,) = xr.broadcast(dso)
    arro = dso.to_array("mlvar", name="mlo")
    arro = arro.transpose("batch", "lev", "mlvar")

    return arr, arro
def countNCdir(filelist: list):

    counter = 0

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
        ds, dso = pad_and_stack_layers_and_vars_1d(ds, dso)

        num_samples = ds.shape[0]
        counter += num_samples


    return counter

def getDataPaths(stride_sample=19):
    #original stride sample was 37
    #because here I only use 5 years of simulation (instead of 10), i wanna use about two times the data then
    f_mli1 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.000[1234]-*-*-*.nc')
    f_mli2 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0005-01-*-*.nc')
    f_mli = [*f_mli1, *f_mli2]
    f_mli = f_mli[::stride_sample]

    # validation dataset for HPO
    f_mli1 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0005-0[23456789]-*-*.nc')
    f_mli2 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0005-1[012]-*-*.nc')
    f_mli3 = glob.glob('../climsim-dataset/ClimSim_low-res/train/*/E3SM-MMF.mli.0006-01-*-*.nc')
    f_mli_val = [*f_mli1, *f_mli2, *f_mli3]
    f_mli_val = f_mli_val[::stride_sample]

    return f_mli, f_mli_val

f_mli, f_mli_val = getDataPaths(stride_sample=19)

print("train: " + str(countNCdir(f_mli)))           #2124672  *19
print("validate: " + str(countNCdir(f_mli_val)))   # 531456   *19
#10091520

