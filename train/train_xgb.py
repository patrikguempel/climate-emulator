import os

import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Loading Data...")

train_x = np.load("train/npy_data/train_input.npy")
train_y = np.load("train/npy_data/train_target.npy")
val_x = np.load("train/npy_data/val_input.npy")
val_y = np.load("train/npy_data/val_target.npy")

dtrain = xgb.DMatrix(train_x, label=train_y)
dval = xgb.DMatrix(val_x, label=val_y)

params = {
    'objective': 'reg:squarederror',            #Used for regression
    'learning_rate': 0.01,
    #'subsample': 0.6,                           # subsampling
    #'colsample_bytree': 0.6,
    'verbosity': 1,                             #logging
    'tree_method': 'hist',
    'device': 'cuda'                            #make XGB use the GPUs
}

logging.info("Data loaded!")

def cross_evaluation():
    logging.info("Performing Cross Validation...")

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=5000,                        # High amount of rounds
        nfold=5,
        metrics="rmse",                             # Evaluate by rmse
        early_stopping_rounds=50,                   # Stop if performance is not increasing
        seed=42,
        verbose_eval=True                           # log
    )
    logging.info(f"Best Boosting-Round: {len(cv_results)}")
    logging.info(f"Cross-Validation Results: \n{cv_results}")

def train():
    logging.info("Training...")

    evals = [(dtrain, 'train'), (dval, 'eval')]

    for max_depth in [15]:
        params['max_depth'] = max_depth

        logging.info(f"Training model with max-depth {max_depth}...")

        bst = xgb.train(params,
                        dtrain,
                        num_boost_round=5000,
                        evals=evals,
                        early_stopping_rounds=50,
                        callbacks=[
                            EarlyStopping(rounds=50, save_best=True, maximize=False, data_name='eval', metric_name='rmse'),
                        ],
                        verbose_eval=True
        )

        logging.info(f"Saving with max-depth {max_depth}...")
        path = f"./models/xgb/d{max_depth}-l"
        os.makedirs(path, exist_ok=True)
        bst.save_model(f"{path}/model.ubj")
        logging.info("Done!")

    for max_depth in range():
        params['max_depth'] = max_depth
        params['subsample'] = 0.8
        params['colsample_bytree'] = 0.8

        logging.info(f"Training model with max-depth {max_depth} and subsample 0.8...")

        bst = xgb.train(params,
                        dtrain,
                        num_boost_round=5000,
                        evals=evals,
                        early_stopping_rounds=50,
                        callbacks=[
                            EarlyStopping(rounds=50, save_best=True, maximize=False, data_name='eval',
                                          metric_name='rmse'),
                        ],
                        verbose_eval=True
                        )

        logging.info(f"Saving with max-depth {max_depth}...")
        path = f"./models/xgb/d{max_depth}-s80-l"
        os.makedirs(path, exist_ok=True)
        bst.save_model(f"{path}/model.ubj")
        logging.info("Done!")

#cross_evaluation()
train()