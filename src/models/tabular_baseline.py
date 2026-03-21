"""
Tabular (non-DL) baseline using sklearn.
Flattens the time-series window into a feature vector.
Models: LinearRegression (simple), RandomForest (stronger ML baseline).
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def flatten(X: np.ndarray) -> np.ndarray:
    """(N, window, features) → (N, window * features)"""
    return X.reshape(X.shape[0], -1)


def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(flatten(X_train), y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, seed=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(flatten(X_train), y_train)
    return model


def evaluate(model, X, y, name=""):
    preds = model.predict(flatten(X))
    mae  = mean_absolute_error(y, preds)
    mape = mean_absolute_percentage_error(y, preds) * 100
    print(f"[{name}] MAE={mae:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "MAPE": mape, "preds": preds}


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from src.data_pipeline import load_dataset
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_dataset()

    lr  = train_linear(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)

    evaluate(lr, X_val, y_val, "LinearRegression-val")
    evaluate(rf, X_val, y_val, "RandomForest-val")

    save_model(lr, "experiments/results/linear_regression.pkl")
    save_model(rf, "experiments/results/random_forest.pkl")
