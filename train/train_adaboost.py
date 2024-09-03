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

# Load data
train_input = np.load('train/npy_data/train_input.npy')[::3000]
train_output = np.load('train/npy_data/train_target.npy')[::3000]
val_input = np.load('train/npy_data/val_input.npy')[::3000]
val_output = np.load('train/npy_data/val_target.npy')[::3000]
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

# Hyperparameter grid
param_grid = {
    'estimator__n_estimators': [100, 200, 300, 400, 500],
    'estimator__learning_rate': [0.005, 0.008, 0.01, 0.1, 0.5]
}

# GridSearchCV with AdaBoostRegressor wrapped in MultiOutputRegressor
grid_search = GridSearchCV(MultiOutputRegressor(AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=1), random_state=42)),
                           param_grid, cv=ps, verbose=1, scoring='neg_mean_squared_error')

# Train the model with Grid Search
grid_search.fit(X_combined, y_combined)

# Display best parameters and accuracy
logging.info(f'Best Parameters: {grid_search.best_params_}')
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(val_input)
logging.info(f'Test MSE: {mean_squared_error(val_output, y_test_pred)}')

# Save the best model
model_name = "ada1_gridsearch"
path = f"models/{model_name}"
os.makedirs(path, exist_ok=True)

joblib.dump(best_model, f'{path}/model.pkl')
logging.info(f"Best model saved as {path}/model.pkl")
