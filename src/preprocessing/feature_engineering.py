import pandas as pd

def add_features(df: pd.DataFrame):

    print("[INFO] Adding time, lag, and rolling features...")

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)

    # ---- Lag Features ----
    df["load_prev_hour"] = df["load_actual"].shift(1)
    df["load_prev_day"] = df["load_actual"].shift(24)

    # ---- Rolling Means ----
    df["rolling_3h"] = df["load_actual"].rolling(3).mean()
    df["rolling_24h"] = df["load_actual"].rolling(24).mean()

    df = df.dropna()  # Remove initial NaN rows (after lagging)

    print("[INFO] Feature engineering complete.")
    return df
