"""
Data pipeline for UCI Household Electric Power Consumption dataset.
Handles loading, cleaning, feature engineering, and train/val/test splitting.

Two modes:
  1. Full pipeline: loads raw .txt file, cleans, resamples, splits, saves .npy
  2. Fast mode:     if data/processed/*.csv already exist, converts them to .npy
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

RAW_PATH      = os.path.join("data", "raw", "household_power_consumption.txt")
PROCESSED_DIR = os.path.join("data", "processed")

FEATURE_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
]
TARGET_COL = "Global_active_power"


# ── Loading ────────────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True, na_values=["?"], low_memory=False,
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Global_active_power"])
    df = df.ffill()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]       = df["datetime"].dt.hour
    df["dayofweek"]  = df["datetime"].dt.dayofweek
    df["month"]      = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("datetime")
    numeric = df.select_dtypes(include="number")
    hourly = numeric.resample("1h").mean().dropna()
    hourly = hourly.reset_index()
    hourly = add_time_features(hourly)
    return hourly


def make_windows(series: np.ndarray, features: np.ndarray,
                 window: int = 24, horizon: int = 1):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(features[i : i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def temporal_split(df: pd.DataFrame, val_frac=0.1, test_frac=0.1):
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start  = int(n * (1 - test_frac - val_frac))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


# ── CSV → NPY conversion (fast path when CSVs already exist) ──────────────────

def _csvs_to_npy(window: int = 24, horizon: int = 1) -> bool:
    """Convert existing processed CSVs to .npy if .npy files are missing."""
    paths = [os.path.join(PROCESSED_DIR, f)
             for f in ["train.csv", "val.csv", "test.csv"]]
    if not all(os.path.exists(p) for p in paths):
        return False

    print("[pipeline] Found existing CSVs — converting to .npy format …")

    def _load(path):
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            # reconstruct from Date + Time if split differently
            df["datetime"] = pd.to_datetime(df.iloc[:, 0])
        else:
            df["datetime"] = pd.to_datetime(df["datetime"])
        if "hour_sin" not in df.columns:
            df = add_time_features(df)
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0.0
        return df

    train_df = _load(paths[0])
    val_df   = _load(paths[1])
    test_df  = _load(paths[2])

    scaler = StandardScaler()
    train_df = train_df.copy()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df  = val_df.copy()
    val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
    test_df = test_df.copy()
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    def _xy(df):
        return make_windows(df[TARGET_COL].values, df[FEATURE_COLS].values,
                            window, horizon)

    X_tr, y_tr = _xy(train_df)
    X_va, y_va = _xy(val_df)
    X_te, y_te = _xy(test_df)

    for name, arr in [("X_train", X_tr), ("y_train", y_tr),
                      ("X_val",   X_va), ("y_val",   y_va),
                      ("X_test",  X_te), ("y_test",  y_te)]:
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), arr)

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"[pipeline] Train={X_tr.shape}  Val={X_va.shape}  Test={X_te.shape}")
    return True


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_dataset(window: int = 24, horizon: int = 1, seed: int = 42):
    """Full pipeline from raw .txt file. Falls back to CSV conversion."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_PATH):
        if _csvs_to_npy(window, horizon):
            return load_dataset()
        raise FileNotFoundError(
            f"Raw data not found at '{RAW_PATH}'.\n"
            "Run: python data/get_data.py"
        )

    print("[pipeline] Loading raw data …")
    df = load_raw()
    df = clean(df)
    df = resample_hourly(df)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    train_df, val_df, test_df = temporal_split(df)

    scaler = StandardScaler()
    train_df = train_df.copy()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df  = val_df.copy()
    val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
    test_df = test_df.copy()
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    X_tr, y_tr = make_windows(train_df[TARGET_COL].values,
                               train_df[FEATURE_COLS].values, window, horizon)
    X_va, y_va = make_windows(val_df[TARGET_COL].values,
                               val_df[FEATURE_COLS].values, window, horizon)
    X_te, y_te = make_windows(test_df[TARGET_COL].values,
                               test_df[FEATURE_COLS].values, window, horizon)

    for name, arr in [("X_train", X_tr), ("y_train", y_tr),
                      ("X_val",   X_va), ("y_val",   y_va),
                      ("X_test",  X_te), ("y_test",  y_te)]:
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), arr)

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"[pipeline] Train={X_tr.shape}  Val={X_va.shape}  Test={X_te.shape}")
    print(f"[pipeline] Saved to {PROCESSED_DIR}/")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, scaler


def load_dataset():
    """Load processed .npy splits. Auto-converts from CSVs if needed."""
    npy = ["X_train.npy", "y_train.npy", "X_val.npy",
           "y_val.npy",   "X_test.npy",  "y_test.npy"]
    missing = [f for f in npy
               if not os.path.exists(os.path.join(PROCESSED_DIR, f))]
    if missing:
        if not _csvs_to_npy():
            raise FileNotFoundError(
                "Processed data not found. Run: python run.py"
            )

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    scaler = pickle.load(open(scaler_path, "rb")) if os.path.exists(scaler_path) else None

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


if __name__ == "__main__":
    build_dataset()
