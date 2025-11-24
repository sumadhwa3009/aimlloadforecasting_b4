import joblib
import json
import os
from tensorflow.keras.models import load_model

def save_lstm(model, scaler, feature_names):
    os.makedirs("saved_models", exist_ok=True)

    # 1. Save LSTM model
    model.save("saved_models/lstm_model.h5")

    # 2. Save scaler
    joblib.dump(scaler, "saved_models/scaler.pkl")

    # 3. Save feature order
    with open("saved_models/features.json", "w") as f:
        json.dump(feature_names, f)

    print("[INFO] LSTM model, scaler & features saved successfully!")
