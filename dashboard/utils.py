import pandas as pd
import numpy as np
import joblib
import json

# Load feature list

def load_feature_names():
    """Load training feature order from saved file."""
    with open("../saved_models/features.json") as f:
        return json.load(f)


# Preprocess ML input

def preprocess_ml_input(df):
    """
    Preprocess input for ML models (XGB, RF, CatBoost, SVM, GB).
    ML models use raw features (no sequences).
    """
    feature_names = load_feature_names()

    # Check missing columns
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV missing required columns: {missing}")

    # Return only feature columns
    return df[feature_names]

# Preprocess DL input

def load_dl_scalers(model_name):
    """
    Load scaler_X and scaler_y for LSTM/GRU/CNN/CNN_LSTM/BPNN.
    Saved as: <model_name>_scaler_X.pkl and <model_name>_scaler_y.pkl
    """
    scaler_X = joblib.load(f"../saved_models/{model_name}_scaler_X.pkl")
    scaler_y = joblib.load(f"../saved_models/{model_name}_scaler_y.pkl")
    return scaler_X, scaler_y


def preprocess_dl_input(df, model_name, lookback=24):
    """
    Preprocess uploaded CSV for deep learning models.
    Steps:
      1. Load feature names
      2. Ensure all required features exist
      3. Load DL scalers
      4. Scale input
      5. Create sequences (24-hour windows)
    """
    feature_names = load_feature_names()

    # Check missing columns
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV missing required columns: {missing}")

    X = df[feature_names]

    # Load scalers
    scaler_X, scaler_y = load_dl_scalers(model_name)

    # Scale inputs
    X_scaled = scaler_X.transform(X)

    # Create sequences
    X_seq = create_sequences(X_scaled, lookback)

    return X_seq, scaler_y

# Sequence creator

def create_sequences(X, lookback=24):
    """
    Convert 2D data → 3D sequence data for LSTM/GRU/CNN/CNN-LSTM.
    Shape becomes: (samples, lookback, features)
    """
    sequences = []
    for i in range(lookback, len(X)):
        sequences.append(X[i - lookback:i])
    return np.array(sequences)
