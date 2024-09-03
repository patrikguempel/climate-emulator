import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def grid_search():
    train_input = np.load('train/npy_data/train_input.npy')[::3000]
    train_output = np.load('train/npy_data/train_target.npy')[::3000]

    # validation data
    val_input = np.load('train/npy_data/val_input.npy')[::3000]
    val_output = np.load('train/npy_data/val_target.npy')[::3000]

    logging.info("Data loaded!")

    train_indices = np.full(train_input.shape[0], 0)  # Kennzeichne alle Trainingsdaten mit 0
    val_indices = np.full(val_input.shape[0], -1)     # Kennzeichne alle Validierungsdaten mit -1
    split_indices = np.concatenate((train_indices, val_indices))  # Kombiniere Indizes

    # Kombinieren von Trainings- und Validierungsdaten
    X_combined = np.concatenate((train_input, val_input), axis=0)
    y_combined = np.concatenate((train_output, val_output), axis=0)

    # Erstellen des PredefinedSplit
    ps = PredefinedSplit(test_fold=split_indices)

    param_grid = {
        'n_estimators': [100, 200, 500, 750, 1000],
        'max_depth': [None, 10, 30, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    logging.info("Performing grid_search...")

    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=ps, verbose=1, scoring='neg_mean_squared_error')

    # Modell trainieren
    grid_search.fit(X_combined, y_combined)

    # Bestes Modell anzeigen
    logging.info(f'Best Parameters: {grid_search.best_params_}')
    best_model = grid_search.best_estimator_

    # 4. Modell auf Testdaten evaluieren
    y_test_pred = best_model.predict(val_input)
    logging.info(f'Test MSE: {mean_squared_error(val_output, y_test_pred)}')

    # 5. Bestes Modell speichern
    model_name = "rf1_gridsearch"
    path = f"models/{model_name}"
    os.makedirs(path, exist_ok=True)

    joblib.dump(best_model, f'{path}/model.pkl')
    logging.info(f"Best model saved as {path}/model.pkl")

def train():
    train_input = np.load('train/npy_data/train_input.npy')
    train_output = np.load('train/npy_data/train_target.npy')

    # validation data
    val_input = np.load('train/npy_data/val_input.npy')
    val_output = np.load('train/npy_data/val_target.npy')

    logging.info("Data loaded!")

    rf_model = RandomForestRegressor(
        max_depth=None,
        min_samples_leaf=4,
        min_samples_split=2,
        n_estimators=750,
        random_state=42
    )

    rf_model.fit(train_input, train_output)

    y_val_pred = rf_model.predict(val_input)
    mse = mean_squared_error(val_output, y_val_pred)
    logging.info(f"Mean Squared Error auf den Validierungsdaten: {mse}")


    joblib.dump(rf_model, 'models/final_rf_model.pkl')
    logging.info("Modell wurde gespeichert!")

grid_search()