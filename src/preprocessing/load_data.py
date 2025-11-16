import pandas as pd

# Final required columns (our universal template)
REQUIRED_COLUMNS = [
    "timestamp",
    "load_actual",
    "temperature",
    "humidity",
    "dew_point",
    "solar_generation",
    "wind_generation"
]

def load_dataset(path: str):
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Strip unwanted spaces
    df.columns = df.columns.str.strip()

    # ---- 1. Standardize column names based on your dataset ----
    rename_map = {
        "time": "timestamp",
        "temperature": "temperature",
        "dew_point": "dew_point",
        "relative_humidity": "humidity",
        "ES_load_actual_entsoe_transparency": "load_actual",
        "ES_solar_generation_actual": "solar_generation",
        "ES_wind_onshore_generation_actual": "wind_generation"
    }

    df = df.rename(columns=rename_map)

    # ---- 2. Keep ONLY required features ----
    df = df[[
        "timestamp",
        "load_actual",
        "temperature",
        "humidity",
        "dew_point",
        "solar_generation",
        "wind_generation"
    ]]

    print("[INFO] Standardized columns:", df.columns.tolist())
    print("[INFO] Rows:", df.shape[0])

    return df
