import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame):
    """
    Adds calendar, lag, and rolling statistical features
    for time-series load forecasting.

    Assumptions:
    - df contains 'timestamp' (datetime)
    - df contains 'load_actual' (target)
    """

    print("[INFO] Adding calendar, lag, and rolling features...")

    # ---- Ensure datetime ----
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Calendar Features ----
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ---- Cyclical Encoding (IMPORTANT) ----
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ---- Lag Features ----
    df["load_lag_1"] = df["load_actual"].shift(1)
    df["load_lag_24"] = df["load_actual"].shift(24)
    df["load_lag_168"] = df["load_actual"].shift(168)

    # ---- Rolling Statistical Features ----
    df["rolling_mean_24"] = df["load_actual"].rolling(window=24).mean()
    df["rolling_std_24"] = df["load_actual"].rolling(window=24).std()
    df["rolling_mean_168"] = df["load_actual"].rolling(window=168).mean()

    # ---- Drop rows with NaN introduced by lag/rolling ----
    df = df.dropna().reset_index(drop=True)

    print("[INFO] Feature engineering complete.")

    return df
