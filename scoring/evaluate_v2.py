
import xarray as xr
import numpy as np
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

data = data_utils(grid_info, mli_mean, mli_max, mli_min, mlo_scale)
data.set_to_v2_vars()



def createMetricsV2(model_name: str):

    data.input_scoring = np.load("scoring/inputs_v2.npy")
    data.target_scoring = np.load("scoring/true_outputs_v2.npy")

    pred_path = f"models/{model_name}/evaluation-v2/predictions.npy"

    data.set_pressure_grid(data_split = 'scoring')

    data.model_names = [model_name]
    data.preds_scoring = dict(zip(data.model_names, [data.load_npy_file(pred_path)]))

    data.reweight_target(data_split='scoring')
    data.reweight_preds(data_split='scoring')

    data.metrics_names = ['MAE', 'RMSE', 'R2']
    data.create_metrics_df(data_split='scoring')


