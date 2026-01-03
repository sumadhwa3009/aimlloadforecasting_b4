import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults

from utils import (
    preprocess_ml_input,
    preprocess_dl_input,
    load_feature_names,
)

LOOKBACK = 24

# Main model loader
def load_model_by_name(model_name, df):
    """
    Unified interface to load & run predictions for:
        ML models → XGBoost, CatBoost, RF, SVM, GradientBoosting
        DL models → LSTM, GRU, CNN, CNN-LSTM, BPNN
        ARIMA (univariate)
    """

    model_name_lower = model_name.lower()

    # ----- ML MODELS ------------
    if model_name in ["XGBoost", "CatBoost", "RandomForest", "SVM", "GradientBoosting"]:

        X_test = preprocess_ml_input(df)
        
        # — XGBoost —
        if model_name == "XGBoost":
            model = xgb.XGBRegressor()
            model.load_model("../saved_models/xgb_model.json")
            preds = model.predict(X_test)

        # — CatBoost —
        elif model_name == "CatBoost":
            model = CatBoostRegressor()
            model.load_model("../saved_models/catboost_model.cbm")
            preds = model.predict(X_test)

        # — RandomForest —
        elif model_name == "RandomForest":
            model = joblib.load("../saved_models/rf_model.pkl")
            preds = model.predict(X_test)

        # — SVM —
        elif model_name == "SVM":
            model = joblib.load("../saved_models/svm_model.pkl")
            preds = model.predict(X_test)

        # — Gradient Boosting —
        elif model_name == "GradientBoosting":
            model = joblib.load("../saved_models/gbr_model.pkl")
            preds = model.predict(X_test)

        return pd.DataFrame({"Predicted Load": preds})

    # ----- DL MODELS ------------
    elif model_name in ["LSTM", "GRU", "CNN", "CNN_LSTM", "BPNN"]:

        model_path = f"../saved_models/{model_name_lower}_model.h5"

        # Special case: BPNN (no sequences)
        if model_name == "BPNN":
            X_test = preprocess_ml_input(df)

            scaler_X = joblib.load("../saved_models/bpnn_scaler_X.pkl")
            scaler_y = joblib.load("../saved_models/bpnn_scaler_y.pkl")

            X_scaled = scaler_X.transform(X_test)

            model = load_model(model_path, compile=False)
            pred_scaled = model.predict(X_scaled)
            preds = scaler_y.inverse_transform(pred_scaled)

            return pd.DataFrame({"Predicted Load": preds.reshape(-1)})

        # LSTM / GRU / CNN / CNN-LSTM
        else:
            X_seq, scaler_y = preprocess_dl_input(df, model_name_lower, lookback=LOOKBACK)

            model = load_model(model_path, compile=False)
            pred_scaled = model.predict(X_seq)
            preds = scaler_y.inverse_transform(pred_scaled)

            # Align with sequence trimming (first 24 timestamps have no prediction)
            pad = [None] * LOOKBACK
            final_preds = pad + preds.reshape(-1).tolist()

            return pd.DataFrame({"Predicted Load": final_preds})

    # ----- ARIMA ----------------
    elif model_name == "ARIMA":
        """
        ARIMA is univariate → only uses load_actual column.
        For uploaded CSV without load_actual, prediction is not possible.
        """

        if "load_actual" not in df.columns:
            raise ValueError("To use ARIMA, uploaded CSV must contain 'load_actual' column.")

        # Load ARIMA
        arima_model = ARIMAResults.load("../saved_models/arima_model.pkl")

        # Forecast for same number of timestamps
        steps = len(df)
        preds = arima_model.forecast(steps=steps)

        return pd.DataFrame({"Predicted Load": preds.values})

    # Unknown model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
