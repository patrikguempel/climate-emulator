import numpy as np
import xgboost as xgb


modelName = "xgb1"

train_x = np.load("train/npy_data/train_input.npy")
train_y = np.load("train/npy_data/train_target.npy")
val_x = np.load("train/npy_data/val_input.npy")
val_y = np.load("train/npy_data/val_target.npy")

dtrain = xgb.DMatrix(train_x, label=train_y)
dval = xgb.DMatrix(val_x, label=val_y)

params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'reg:squarederror'
}

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,  # Eine hohe Anzahl an Runden
    nfold=5,
    metrics="rmse",  # Metrik, um die Leistung zu bewerten
    early_stopping_rounds=50,  # Stoppen, wenn sich die Leistung nicht verbessert
    as_pandas=True,
    seed=42
)

# Beste Anzahl von Runden basierend auf Cross-Validation
best_num_rounds = cv_results.shape[0]

evals = [(dtrain, 'train'), (dval, 'eval')]

bst = xgb.train(params,
                dtrain,
                best_num_rounds,
                evals=evals,
                early_stopping_rounds=50)

bst.save_model(f"./models/{modelName}/xgb_model.model")
print("Done!")

