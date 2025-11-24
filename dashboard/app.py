import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import load_model

#from utils import preprocess_input, create_sequences
from model_loader import load_model_by_name

st.set_page_config(page_title="Load Forecasting Dashboard", layout="wide")

# ===========================
# Global UI Styling
# ===========================
st.markdown("""
<style>

body {
    font-family: 'Segoe UI', sans-serif;
}

/* Title Styling */
h1 {
    color: #4CAF50;
    font-weight: 800;
}

/* Subheaders */
h2, h3, h4 {
    color: #2E8B57;
    margin-top: 25px;
}

/* Card Boxes */
.st-card {
    padding: 20px;
    background: #f8f9fa;
    border-radius: 12px;
    border: 1px solid #dcdcdc;
    margin-bottom: 20px;
}

/* Center Text */
.center-text {
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1e1e1e;
    color: white;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
}

</style>
""", unsafe_allow_html=True)


st.sidebar.title("Load Forecasting Dashboard")

page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Upload & Predict", "Compare Models", "Visualizations", "About"]
)

if page == "Home":
    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    st.title("Load Forecasting System Dashboard")

    st.markdown("""
    <div class='center-text'>
    A unified system for training, comparing, and forecasting energy load  
    using Machine Learning, Deep Learning, and Statistical Models.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    st.subheader("Features of this Dashboard")

    st.markdown("""
    - Upload data & predict using **11 models**  
    - Compare models side-by-side  
    - Explore powerful visualizations  
    - Integrated ARIMA, ML, DL forecasting  
    - Clean & intuitive UI  
    """)
    st.markdown('</div>', unsafe_allow_html=True)


if page == "Upload & Predict":
    st.title("Upload & Predict")

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:")
        st.dataframe(df.head())

        model_name = st.selectbox(
            "Select Model for Prediction",
            ["XGBoost", "CatBoost", "RandomForest", "SVM",
             "GradientBoosting", "LSTM", "GRU", "CNN",
             "CNN_LSTM", "BPNN", "ARIMA"]
        )

        if st.button("Predict"):
            with st.spinner("Predicting... Please wait"):
                preds = load_model_by_name(model_name, df)

            st.success("Prediction Complete!")
            st.write(preds.head())

            st.line_chart(preds)

if page == "Compare Models":
    st.title("Compare All Models")

    st.markdown("""
    This page shows the performance of all trained models
    using **MAE**, **RMSE**, and **MAPE**.
    
    The table is auto-loaded from:
    `results/model_comparison.csv`
    """)

    try:
        results_df = pd.read_csv("../notebooks/results/model_comparison.csv")
    except:
        st.error("Comparison file not found! Please run model comparison first.")
        st.stop()

    # Display ranked table
    st.subheader("Ranked Performance Table")
    st.dataframe(results_df, use_container_width=True)

    # Best model
    best_model = results_df.iloc[0]

    st.success(f"""
    ### Best Model: **{best_model['Model']}**

    - **MAPE:** {best_model['MAPE']:.3f}%  
    - **MAE:** {best_model['MAE']:.2f}  
    - **RMSE:** {best_model['RMSE']:.2f}
    """)



    # ===== BAR CHARTS =====
    st.subheader("Error Metrics Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### MAPE (%)")
        st.bar_chart(results_df.set_index("Model")["MAPE"])

    with col2:
        st.write("### MAE")
        st.bar_chart(results_df.set_index("Model")["MAE"])

    with col3:
        st.write("### RMSE")
        st.bar_chart(results_df.set_index("Model")["RMSE"])

    # ===== DOWNLOAD OPTION =====
    st.subheader("⬇Download Comparison Table")
    st.download_button(
        label="Download CSV",
        data=results_df.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv"
    )

if page == "Visualizations":
    st.title("Data Visualizations")

    st.markdown("""
    Explore patterns and relationships in the dataset.
    All plots auto-update based on your uploaded or default dataset.
    """)

    # Load processed dataset
    try:
        df_viz = pd.read_csv("../data/processed/final_dataset2.csv", parse_dates=["timestamp"])
    except:
        st.error("Could not load dataset for visualization.")
        st.stop()

    # 1. Time Series Trend
    st.subheader("Load Trend Over Time")
    st.line_chart(df_viz.set_index("timestamp")["load_actual"])


    # 2. Hourly Load Pattern

    st.subheader("Hourly Pattern (0-23)")

    df_viz["hour"] = df_viz["timestamp"].dt.hour
    hourly_avg = df_viz.groupby("hour")["load_actual"].mean()

    st.bar_chart(hourly_avg)

    # 3. Daily Pattern (Mon–Sun)
    st.subheader("Daily Pattern (Mon-Sun)")

    df_viz["day_name"] = df_viz["timestamp"].dt.day_name()
    daily_avg = df_viz.groupby("day_name")["load_actual"].mean()

    # Sort weekday order
    week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_avg = daily_avg.reindex(week_order)

    st.bar_chart(daily_avg)

    # 4. Correlation Heatmap
    st.subheader("Correlation Heatmap")

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_viz.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 5. Weather vs Load (Scatter Plots)
    st.subheader("Weather vs Load Correlation")

    weather_cols = ["temperature", "humidity", "dew_point", "solar_generation", "wind_generation"]

    for col in weather_cols:
        if col in df_viz.columns:
            st.write(f"### {col} vs Load")
            fig2, ax2 = plt.subplots()
            ax2.scatter(df_viz[col], df_viz["load_actual"], alpha=0.3)
            ax2.set_xlabel(col)
            ax2.set_ylabel("Load Actual")
            st.pyplot(fig2)

    # 6. Moving Average Trends
    st.subheader("Moving Average Trends")

    df_viz["MA7"] = df_viz["load_actual"].rolling(7*24).mean()
    df_viz["MA30"] = df_viz["load_actual"].rolling(30*24).mean()

    st.line_chart(df_viz.set_index("timestamp")[["load_actual", "MA7", "MA30"]])

    st.success("Visualization Completed ✔")

if page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Load Forecasting Using Advanced Machine Learning & Deep Learning Models

    This dashboard is part of a complete end-to-end **Load Forecasting System**
    developed using classical ML, DL, AutoML, and statistical models.

    The system predicts electrical load accurately by learning from:

    - Weather data  
    - Past load values  
    - Time-based features  
    - Renewable energy generation  

    ---
    """)

    st.header("Project Workflow")

    st.markdown("""
    ### **1. Data Collection**
    - Raw load and weather data collected (open-source sources like ENTSO-E).
    - Additional fields included: temperature, humidity, solar, wind, dew point.

    ### **2. Data Preprocessing**
    - Missing values handled using interpolation.
    - Outlier smoothing.
    - Data augmentation applied to reduce noise.
    - Timestamp converted to features:
        - Hour, Day, Month, Day of Week, Weekend flag.
    - Final dataset saved as:  
      **`data/processed/final_dataset2.csv`**

    ### **3. Feature Engineering**
    - Lag features (t-1, t-24, t-48)
    - Rolling windows (mean, std)
    - Time-based encoding
    - Future-proof template created in CSV format  

    ### **4. Model Training**
    Models implemented:

    #### Machine Learning Models
    - Linear Regression  
    - Random Forest  
    - XGBoost  
    - Gradient Boosting  
    - LightGBM  
    - CatBoost  
    - SVM  

    #### Deep Learning Models
    - LSTM  
    - GRU  
    - 1D CNN  
    - CNN-LSTM Hybrid  
    - BPNN (Backprop Neural Network)

    #### Statistical Model
    - ARIMA (Auto-Regressive Integrated Moving Average)

    All models saved inside **`saved_models/`**.

    ### **5. Model Evaluation**
    Metrics used:
    - **MAE (Mean Absolute Error)**
    - **RMSE (Root Mean Square Error)**
    - **MAPE (Mean Absolute Percentage Error)**

    A unified comparison table generated:
    - `results/model_comparison.csv`
    - `results/model_comparison.json`

    ### **6. Ensemble & AutoML**
    - Weighted ensemble attempted (XGB + RF + CatBoost)
    - AutoML tuning using FLAML and Optuna

    ### **7. Dashboard Development**
    Built using **Streamlit** with:
    - Upload & predict functionality
    - Multi-model selection
    - Visualization tools
    - Model comparison UI

    ---
    """)

    st.header("Tools & Technologies Used")

    st.markdown("""
    - **Python 3.x**
    - **Pandas, NumPy**
    - **Scikit-Learn**
    - **TensorFlow / Keras**
    - **XGBoost, CatBoost, LightGBM**
    - **Statsmodels (ARIMA)**
    - **Optuna (Hyperparameter tuning)**
    - **FLAML (AutoML)**
    - **Matplotlib, Seaborn**
    - **Streamlit (Dashboard)**
    """)

    st.header("Future Enhancements")

    st.markdown("""
    - Integrate **real-time load forecasting** (live data stream)
    - Deploy dashboard on **Streamlit Cloud / AWS**
    - Add **LSTM Encoder-Decoder** architecture
    - Try **N-BEATS**, **Temporal Fusion Transformer (TFT)**
    - Add **real-time alerts** for peak load situations
    """)

    st.success("This page documents the entire workflow of your Load Forecasting Project")


