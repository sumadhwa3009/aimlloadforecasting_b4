import pandas as pd


# ---- Mandatory target column ----
TARGET_COLUMN = "load_actual"

# ---- Optional feature columns (may or may not exist in external data) ----
OPTIONAL_COLUMNS = [
    "temperature",
    "humidity",
    "dew_point",
    "solar_generation",
    "wind_generation"
]


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads raw dataset and standardizes column names.
    Ensures compatibility across training, testing, and dashboard inference.
    """

    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)

    # ---- Strip unwanted spaces ----
    df.columns = df.columns.str.strip()

    # ---- Standardize column names (flexible mapping) ----
    rename_map = {
        # Time
        "time": "timestamp",
        "datetime": "timestamp",

        # Load
        "ES_load_actual_entsoe_transparency": "load_actual",
        "load": "load_actual",
        "actual_load": "load_actual",

        # Weather
        "relative_humidity": "humidity",
        "apparent_temperature": "temperature",

        # Renewables
        "ES_solar_generation_actual": "solar_generation",
        "ES_wind_onshore_generation_actual": "wind_generation",
        "wind_generation_actual": "wind_generation",
        "wind_onshore_generation_actual": "wind_generation",
        "solar_generation_actual": "solar_generation"
    }

    df = df.rename(columns=rename_map)

    # ---- Check mandatory columns ----
    if "timestamp" not in df.columns:
        raise ValueError("Mandatory column 'timestamp' not found")

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Mandatory target column 'load_actual' not found")

    # ---- Ensure optional columns exist ----
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # ---- Final column order ----
    final_columns = ["timestamp", TARGET_COLUMN] + OPTIONAL_COLUMNS
    df = df[final_columns]

    print("[INFO] Final standardized columns:", df.columns.tolist())
    print("[INFO] Rows loaded:", df.shape[0])

    return df
