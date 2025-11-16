import pandas as pd
import numpy as np

def clean_and_augment(df: pd.DataFrame):

    print("[INFO] Cleaning dataset...")

    # 1. Convert to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # 2. Replace blank values
    df.replace(["", " ", None], np.nan, inplace=True)

    numeric_cols = [
        "load_actual", "temperature", "humidity",
        "dew_point", "solar_generation", "wind_generation"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # ⭐ FINAL FIX: Use linear interpolation instead of time
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    print("[INFO] Linear interpolation applied successfully.")

    # 3. Augmentation
    print("[INFO] Creating augmented dataset...")

    df_aug = df.copy()
    for col in numeric_cols:
        noise = np.random.normal(0, 0.01, len(df))
        df_aug[col] = df_aug[col] * (1 + noise)

    df_final = pd.concat([df, df_aug]).reset_index()

    print("[INFO] Final rows after augmentation:", df_final.shape[0])

    return df_final
