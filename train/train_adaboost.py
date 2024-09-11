import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def grid_search():
    # Load data
    train_input = np.load('train/npy_data/train_input.npy')[::100]
    train_output = np.load('train/npy_data/train_target.npy')[::100]
    val_input = np.load('train/npy_data/val_input.npy')[::100]
    val_output = np.load('train/npy_data/val_target.npy')[::100]
    logging.info("Data loaded!")

    # Prepare indices for PredefinedSplit
    train_indices = np.full(train_input.shape[0], 0)
    val_indices = np.full(val_input.shape[0], -1)
    split_indices = np.concatenate((train_indices, val_indices))

    # Combine training and validation data
    X_combined = np.concatenate((train_input, val_input), axis=0)
    y_combined = np.concatenate((train_output, val_output), axis=0)

    # Create PredefinedSplit
    ps = PredefinedSplit(test_fold=split_indices)

    for max_depth in [15]:
        # Hyperparameter grid
        param_grid = {
            'estimator__n_estimators': [500, 1000, 2000],
            'estimator__learning_rate': [0.01]
        }

        # GridSearchCV with AdaBoostRegressor wrapped in MultiOutputRegressor
        grid_search = GridSearchCV(MultiOutputRegressor(AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=max_depth), random_state=42)),
                                   param_grid, cv=ps, verbose=1, scoring='neg_mean_squared_error')

        # Train the model with Grid Search
        grid_search.fit(X_combined, y_combined)

        # Display best parameters and accuracy
        logging.info(f'Best Parameters for max depth {max_depth}: {grid_search.best_params_}')
        best_model = grid_search.best_estimator_
        y_test_pred = best_model.predict(val_input)
        logging.info(f'Test MSE: {mean_squared_error(val_output, y_test_pred)}')

def train():
    train_input = np.load('train/npy_data/train_input.npy')
    train_output = np.load('train/npy_data/train_target.npy')
    val_input = np.load('train/npy_data/val_input.npy')
    val_output = np.load('train/npy_data/val_target.npy')
    logging.info("Data loaded!")

    model = MultiOutputRegressor(
        AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=400,
            learning_rate=0.01,
            random_state=42,
        )
    )

    model.fit(train_input, train_output)
    mse = mean_squared_error(val_input, val_output)

    print(f"Mean Squared Error: {mse:.4f}")

    modelName = "ada1"
    path = f"models/{modelName}"
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, f"{path}/model.joblib")
    logging.info("Saved!")

train()