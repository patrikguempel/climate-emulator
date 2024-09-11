import glob
import os
import random

import joblib
import tensorflow as tf
import xarray as xr
import numpy as np
import pandas as pd
import xgboost as xgb

from tensorflow.keras import backend as K

from data_utils import data_utils
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

mli_mean = xr.open_dataset('norm/input_mean.nc')
mli_min = xr.open_dataset('norm/input_min.nc')
mli_max = xr.open_dataset('norm/input_max.nc')
mlo_scale = xr.open_dataset('norm/output_scale.nc')
grid_path = 'grid_info/ClimSim_low-res_grid-info.nc'
grid_info = xr.open_dataset(grid_path)

def makePredictionsRF(modelName: str):
    modelPath = f"models/{modelName}"
    model = joblib.load(modelPath + "/model.joblib")
    logging.info("Model loaded!")

    inputs = np.load("scoring/inputs.npy")
    logging.info("Scoring Data loaded!")

    predictions = model.predict(inputs)
    logging.info("Predictions made!")

    evaluationPath = f"{modelPath}/evaluation"
    os.makedirs(evaluationPath, exist_ok=True)

    np.save(f"{evaluationPath}/predictions.npy", predictions)
    logging.info("Predictions saved!")



def makePredictionsMLP(modelName: str):
    modelPath = f"models/{modelName}"
    model = tf.keras.models.load_model(f"{modelPath}/model.h5")
    logging.info("Model loaded!")

    inputs = np.load("scoring/inputs.npy")
    logging.info("Scoring Data loaded!")

    predictions = model.predict(inputs)
    logging.info("Predictions made!")

    evaluationPath = f"{modelPath}/evaluation"
    os.makedirs(evaluationPath, exist_ok=True)

    np.save(f"{evaluationPath}/predictions.npy", predictions)
    logging.info("Predictions saved!")


def makePredictionsCNN(modelName: str):
    def mae_adjusted(y_true, y_pred):
        ae = K.abs(y_pred - y_true)
        return K.mean(ae[:, :, 0:2]) * (120 / 128) + K.mean(ae[:, :, 2:10]) * (8 / 128)

    def mse_adjusted(y_true, y_pred):
        se = K.square(y_pred - y_true)
        return K.mean(se[:, :, 0:2]) * (120 / 128) + K.mean(se[:, :, 2:10]) * (8 / 128)

    def continuous_ranked_probability_score(y_true, y_pred):
        score = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)), axis=-1)
        diff = tf.subtract(tf.expand_dims(y_pred, -1), tf.expand_dims(y_pred, -2))
        score = tf.add(score,
                       tf.multiply(tf.constant(-0.5, dtype=diff.dtype), tf.reduce_mean(tf.abs(diff), axis=(-2, -1))))

        return tf.reduce_mean(score)

    modelPath = f"models/{modelName}"
    model = tf.keras.models.load_model(f"{modelPath}/model.h5", custom_objects={'mae_adjusted': mae_adjusted,
                                                                                'mse_adjusted': mse_adjusted,
                                                                                'continuous_ranked_probability_score': continuous_ranked_probability_score})
    logging.info("Model loaded!")

    inputs = np.load("scoring/inputs.npy")
    logging.info("Scoring Data loaded!")

    logging.info("Reshaping dataset for CNN...")
    data = data_utils(grid_info=grid_info,
                      input_mean=mli_mean,
                      input_max=mli_max,
                      input_min=mli_min,
                      output_scale=mlo_scale)

    data.set_to_v1_vars()

    inputs_cnn = data.reshape_input_for_cnn(inputs)
    logging.info("Dataset reshaped! Making predictions...")

    predictions_cnn = model.predict(inputs_cnn)
    predictions = data.reshape_target_from_cnn(predictions_cnn)
    logging.info("Predictions made (and reshaped)!")

    evaluationPath = f"{modelPath}/evaluation"
    os.makedirs(evaluationPath, exist_ok=True)

    np.save(f"{evaluationPath}/predictions.npy", predictions)
    logging.info("Data saved!")


def makePredictionsXGB(modelName: str):
    xgp_path = f"models/xgb/{modelName}"

    bst = xgb.Booster({'predictor': 'gpu_predictor'})
    bst.load_model(f"{xgp_path}/model.ubj")
    logging.info("Model loaded!")

    inputs = np.load("scoring/inputs.npy")
    inputs_xgb = xgb.DMatrix(inputs)
    logging.info("Scoring Data loaded!")

    logging.info("Making predictions...")
    predictions = bst.predict(inputs_xgb)
    logging.info("Predictions made!")

    evaluationPath = f"{xgp_path}/evaluation"
    os.makedirs(evaluationPath, exist_ok=True)

    np.save(f"{evaluationPath}/predictions.npy", predictions)
    logging.info("Data saved!")



def createMetricsCSV(model_name: str, evaluationPath = None):
    logging.info("Calculating Metrics CSV...")
    scoringPath = "scoring"
    if evaluationPath == None:
        evaluationPath = f"models/{model_name}/evaluation"

    fn_x_true = f"{scoringPath}/inputs.npy"  # inputs
    fn_y_true = f"{scoringPath}/true_outputs.npy"  # outputs
    fn_y_pred = f"{evaluationPath}/predictions.npy"  # predictions

    # model grid information (nc)
    #fn_grid = './grid_info/ClimSim_low-res_grid-info.nc'
    #fn_grid = grid_info

    # normalization scale factors (nc)
    fn_mli_mean = 'norm/input_mean.nc'
    fn_mli_min = 'norm/input_min.nc'
    fn_mli_max = 'norm/input_max.nc'
    fn_mlo_scale = 'norm/output_scale.nc'

    # fn_save_output
    fn_save_metrics = f'{evaluationPath}/metrics.csv'
    fn_save_metrics_avg = f'{evaluationPath}/metrics.lev-avg.csv'

    # --------------------------------#

    # physical constants from (E3SM_ROOT/share/util/shr_const_mod.F90)
    grav = 9.80616  # acceleration of gravity ~ m/s^2
    cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
    lv = 2.501e6  # latent heat of evaporation ~ J/kg
    lf = 3.337e5  # latent heat of fusion      ~ J/kg
    ls = lv + lf  # latent heat of sublimation ~ J/kg
    rho_air = 101325. / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15  # density of dry air at STP  ~ kg/m^3
    # ~ 1.2923182846924677
    # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
    # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
    # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
    rho_h20 = 1.e3  # density of fresh water     ~ kg/m^ 3

    vars_mlo_energy_conv = {'ptend_t': cp,
                            'ptend_q0001': lv,
                            'cam_out_NETSW': 1.,
                            'cam_out_FLWDS': 1.,
                            'cam_out_PRECSC': lv * rho_h20,
                            'cam_out_PRECC': lv * rho_h20,
                            'cam_out_SOLS': 1.,
                            'cam_out_SOLL': 1.,
                            'cam_out_SOLSD': 1.,
                            'cam_out_SOLLD': 1.
                            }
    vars_longname = \
        {'ptend_t': 'Heating tendency, ∂T/∂t',
         'ptend_q0001': 'Moistening tendency, ∂q/∂t',
         'cam_out_NETSW': 'Net surface shortwave flux, NETSW',
         'cam_out_FLWDS': 'Downward surface longwave flux, FLWDS',
         'cam_out_PRECSC': 'Snow rate, PRECSC',
         'cam_out_PRECC': 'Rain rate, PRECC',
         'cam_out_SOLS': 'Visible direct solar flux, SOLS',
         'cam_out_SOLL': 'Near-IR direct solar flux, SOLL',
         'cam_out_SOLSD': 'Visible diffused solar flux, SOLSD',
         'cam_out_SOLLD': 'Near-IR diffused solar flux, SOLLD'}

    # --------------------------#
    dim_name_level = 'lev'
    dim_name_sample = 'sample'

    # load input dataset
    x_true = np.load(fn_x_true).astype(np.float64)
    y_true = np.load(fn_y_true).astype(np.float64)
    if fn_y_pred[-3:] == '.h5':
        y_pred = xr.open_dataset(fn_y_pred)['pred'].values
    else:
        y_pred = np.load(fn_y_pred).astype(np.float64)
    N_samples = y_pred.shape[0]

    # load norm/scale factors
    mlo_scale = xr.open_dataset(fn_mlo_scale)
    mli_mean = xr.open_dataset(fn_mli_mean)
    mli_min = xr.open_dataset(fn_mli_min)
    mli_max = xr.open_dataset(fn_mli_max)

    # load grid information
    ds_grid = grid_info  # has ncol:384
    N_ncol = len(ds_grid['ncol'])  # length of ncol dimension (nlat * nlon)

    # make area-weights
    ds_grid['area_wgt'] = ds_grid['area'] / ds_grid['area'].mean('ncol')

    # map ds_grid's ncol dimension -> the N_samples dimension of npy-loayd arrays (e.g., y_pred)
    to_xarray = {'area_wgt': (dim_name_sample, np.tile(ds_grid['area_wgt'], int(N_samples / len(ds_grid['ncol'])))),
                 }
    to_xarray = xr.Dataset(to_xarray)

    # add nsample-mapped grid variables back to ds_grid
    ds_grid = xr.merge([ds_grid[['P0', 'hyai', 'hyam', 'hybi', 'hybm', 'lat', 'lon', 'area']],
                        to_xarray[['area_wgt']]])

    # list of ML output variables
    vars_mlo = ['ptend_t', 'ptend_q0001', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC',
                'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD',
                'cam_out_SOLLD']  # mlo mean ML output.

    # length of each variable
    # (make sure that the order of variables are correct)
    vars_mlo_len = {'ptend_t': 60,
                    'ptend_q0001': 60,
                    'cam_out_NETSW': 1,
                    'cam_out_FLWDS': 1,
                    'cam_out_PRECSC': 1,
                    'cam_out_PRECC': 1,
                    'cam_out_SOLS': 1,
                    'cam_out_SOLL': 1,
                    'cam_out_SOLSD': 1,
                    'cam_out_SOLLD': 1
                    }

    # map the length of dimension to the name of dimension
    len_to_dim = {60: dim_name_level,
                  N_samples: dim_name_sample}

    # Here, we first construct a dictionary of {var name: (dimension name, array-like)},
    # then, map the dictionary to an Xarray Dataset.
    # (ref: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)

    DS = {}

    for kds in ['true', 'pred']:
        if kds == 'true':
            work = y_true
        elif kds == 'pred':
            work = y_pred

        # [1] Construct dictionary for xarray dataset
        #     format is key for variable name /
        #               value for a turple of (dimension names, data).
        to_xarray = {}
        for k, kvar in enumerate(vars_mlo):

            # length of variable (ie, number of levels)
            kvar_len = vars_mlo_len[kvar]

            # set dimensions of variable
            if kvar_len == 60:
                kvar_dims = (dim_name_sample, dim_name_level)
            elif kvar_len == 1:
                kvar_dims = dim_name_sample

            # set start and end indices of variable in the loaded numpy array
            # then, add 'kvar':(kvar_dims, <np_array>) to dictionary
            if k == 0: ind1 = 0
            ind2 = ind1 + kvar_len

            # scaled output
            kvar_data = np.squeeze(work[:, ind1:ind2])
            # unscaled output
            kvar_data = kvar_data / mlo_scale[kvar].values

            to_xarray[kvar] = (kvar_dims, kvar_data)

            ind1 = ind2

        # [2] convert dict to xarray dataset
        DS[kds] = xr.Dataset(to_xarray)

        # [3] add surface pressure ('state_ps') from ml input
        # normalized ps
        state_ps = xr.DataArray(x_true[:, 120], dims=('sample'), name='state_ps')
        # denormalized ps
        state_ps = state_ps * (mli_max['state_ps'] - mli_min['state_ps']) + mli_mean['state_ps']
        DS[kds]['state_ps'] = state_ps

        # [4] add grid information
        DS[kds] = xr.merge([DS[kds], ds_grid])

        # [5] add pressure thickness of each level, dp
        # FYI, in a hybrid sigma vertical coordinate system, pressure at level z is
        # P[x,z] = hyam[z]*P0 + hybm[z]*PS[x,z],
        # where, hyam and hybm are
        tmp = DS[kds]['P0'] * DS[kds]['hyai'] + DS[kds]['state_ps'] * DS[kds]['hybi']
        tmp = tmp.isel(ilev=slice(1, 61)).values - tmp.isel(ilev=slice(0, 60)).values
        tmp = tmp.transpose()
        DS[kds]['dp'] = xr.DataArray(tmp, dims=('sample', 'lev'))

        # [6] break (sample) to (ncol,time)
        N_timestep = int(N_samples / N_ncol)
        dim_ncol = np.arange(N_ncol)
        dim_timestep = np.arange(N_timestep)
        new_ind = pd.MultiIndex.from_product([dim_timestep, dim_ncol],
                                             names=['time', 'ncol'])
        DS[kds] = DS[kds].assign_coords(sample=new_ind).unstack('sample')

    del work, to_xarray, y_true, y_pred, x_true, state_ps, tmp

    # [1] Weight vertical levels by dp/g that is equivalent to a mass of air within a grid cell per unit area [kg/m2]
    # [2] Weight horizontal area of each grid cell by a[x]/mean(a[x]).
    # [3] Unit conversion to a common energy unit

    DS_ENERGY = {}
    for kds in ['true', 'pred']:
        # Make a copy to keep original dataset
        DS_ENERGY[kds] = DS[kds].copy(deep=True)

        # vertical weighting / area weighting / unit conversion
        for kvar in vars_mlo:

            # [1] weight vertical levels by dp/g
            #     ONLY for vertically-resolved variables, e.g., ptend_{t,q0001}
            # dp/g = - \rho * dz
            if vars_mlo_len[kvar] == 60:
                DS_ENERGY[kds][kvar] = DS_ENERGY[kds][kvar] * DS_ENERGY[kds]['dp'] / grav

            # [2] weight area
            #     for ALL variables
            DS_ENERGY[kds][kvar] = DS_ENERGY[kds]['area_wgt'] * DS_ENERGY[kds][kvar]

            # [3] convert units to W/m2
            #     for variables with different units, e.g., ptend_{t,q0001}, precsc, precc
            DS_ENERGY[kds][kvar] = vars_mlo_energy_conv[kvar] * DS_ENERGY[kds][kvar]

    all_metrics = ['MAE', 'RMSE', 'R2']
    # A. Calculate metrics
    # After this step,
    # ptend_{t,q0001} have [ncol, lev] dimension;
    # and the rest variables have [ncol] dimension.

    # if spatial analysis is desired (e.g., R2 distribution on global map or on latitude-level plane),
    # the metrics at this step should be used.

    # Select only ML output varibles
    DS_ENERGY[kds] = DS_ENERGY[kds][vars_mlo]

    # Caclulate 3 metrics
    Metrics = {}
    Metrics['MAE'] = (np.abs(DS_ENERGY['true'] - DS_ENERGY['pred'])).mean('time')
    Metrics['RMSE'] = np.sqrt(((DS_ENERGY['true'] - DS_ENERGY['pred']) ** 2.).mean('time'))
    Metrics['R2'] = 1 - ((DS_ENERGY['true'] - DS_ENERGY['pred']) ** 2.).sum('time') / \
                    ((DS_ENERGY['true'] - DS_ENERGY['true'].mean('time')) ** 2.).sum('time')

    # Save grid-wise metric files in netcdf format
    if True:
        os.makedirs(f'{evaluationPath}/gridwise/', exist_ok=True)
        for kmetric in ['MAE', 'RMSE', 'R2']:
            fn_save = f'{evaluationPath}/gridwise/{kmetric}.nc'
            Metrics[kmetric].to_netcdf(fn_save)

    # B. Make horizontal mean.
    # After this step,
    # ptend_{t,q0001} have [lev] dimension;
    # and the rest variables have zero dimensions, i.e., scalars.

    for kmetric in all_metrics:
        Metrics[kmetric] = Metrics[kmetric].mean('ncol')  # simple mean

    # C-1. Save the result after B.
    # to save in a table format as a csv file, the level dimensions are flattened.

    Metrics_stacked = {}
    for kmetric in all_metrics:
        Metrics_stacked[kmetric] = Metrics[kmetric].to_stacked_array('ml_out_idx', sample_dims='', name=kmetric)

    # save the output
    work = pd.DataFrame({'MAE': Metrics_stacked['MAE'].values,
                         'RMSE': Metrics_stacked['RMSE'].values,
                         'R2': Metrics_stacked['R2'].values}
                        )
    work.index.name = 'output_idx'

    # fn_save_metrics = f'./metrics/{model_name}.metrics.csv'
    work.to_csv(fn_save_metrics)

    # C-2. Save the result after vertical averaging.
    # After this step,
    # ptend_{t,q0001} also have zero dimensions, i.e., scalars;

    # Then, the results are saved to a csv file.
    # This csv file will be used for generating plots.

    Metrics_vert_avg = {}
    for kmetric in all_metrics:
        Metrics_vert_avg[kmetric] = Metrics[kmetric].mean('lev')
        Metrics_vert_avg[kmetric] = Metrics_vert_avg[kmetric].mean('ilev')  # remove dummy dim

    # save the output
    work = pd.DataFrame({'MAE': Metrics_vert_avg['MAE'].to_pandas(),
                         'RMSE': Metrics_vert_avg['RMSE'].to_pandas(),
                         'R2': Metrics_vert_avg['R2'].to_pandas()}
                        )
    work.index.name = 'Variable'

    # fn_save_metrics_avg = f'./metrics/{model_name}.metrics.lev-avg.csv'
    work.to_csv(fn_save_metrics_avg)
    logging.info("Done!")


#for modelName in ["d15-s80"]:
#    makePredictionsXGB(modelName)
#    createMetricsCSV(modelName, evaluationPath=f"models/xgb/{modelName}/evaluation")


#for modelName in ["ed", "mlp2"]:
#    makePredictionsMLP(modelName)
#    createMetricsCSV(modelName)

modelName = "mlp_optimized"
makePredictionsMLP(modelName)
createMetricsCSV(modelName)
