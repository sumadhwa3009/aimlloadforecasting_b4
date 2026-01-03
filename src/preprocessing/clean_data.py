import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame):
    """
    Cleans raw load forecasting dataset safely.
    Handles duplicate columns, missing values, and time-aware interpolation.
    """

    print("[INFO] Cleaning dataset...")

    # ---- Remove duplicate columns (CRITICAL FIX) ----
    df = df.loc[:, ~df.columns.duplicated()]

    # ---- Timestamp handling ----
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # ---- Replace empty values ----
    df.replace(["", " ", None], np.nan, inplace=True)

    # ---- Identify numeric columns ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---- Ensure target exists ----
    if "load_actual" not in numeric_cols:
        raise ValueError("Target column 'load_actual' not found in dataset")

    # ---- Convert numeric columns safely (Series-only) ----
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Time-aware interpolation ----
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")

    # ---- Forward & backward fill ----
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # ---- Reset index ----
    df = df.reset_index()

    print("[INFO] Data cleaning complete. Rows:", df.shape[0])

    return df
