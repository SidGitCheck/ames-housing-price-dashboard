import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Ames Housing Price Prediction",
                   layout="wide")

# ================================================================
# Load Artifacts
# ================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    artifacts = {}

    # Load XGBoost
    if os.path.exists("house_price_xgb_model.pkl"):
        artifacts["xgb"] = joblib.load("house_price_xgb_model.pkl")
    else:
        artifacts["xgb"] = None

    # Load LightGBM
    if os.path.exists("house_price_lgb_model.pkl"):
        artifacts["lgb"] = joblib.load("house_price_lgb_model.pkl")
    else:
        artifacts["lgb"] = None

    # Load Scaler
    if os.path.exists("scaler.pkl"):
        artifacts["scaler"] = joblib.load("scaler.pkl")
    else:
        artifacts["scaler"] = None

    # Load training data for insights
    if os.path.exists("train.csv"):
        artifacts["train"] = pd.read_csv("train.csv")

    # Load features order
    if os.path.exists("features.pkl"):
        artifacts["features"] = joblib.load("features.pkl")

    return artifacts


artifacts = load_models()


# ================================================================
# Preprocessing
# ================================================================
def preprocess(df):
    df = df.copy()

    # Select numerical predictors only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = df[numeric_cols]

    # Align with training features
    feature_cols = artifacts.get("features")
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

    # Scale
    scaler = artifacts.get("scaler")
    if scaler:
        df = scaler.transform(df)

    return df


# ================================================================
# Predict function
# ================================================================
def predict_price(df_input):
    df_processed = preprocess(df_input)

    xgb_model = artifacts.get("xgb")
    lgb_model = artifacts.get("lgb")

    pred_xgb = xgb_model.predict(df_processed) if xgb_model else None
    pred_lgb = lgb_model.predict(df_processed) if lgb_model else None

    # Ensemble: Weighted Average
    if pred_xgb is not None and pred_lgb is not None:
        final_pred = (0.6 * np.expm1(pred_xgb)) + (0.4 * np.expm1(pred_lgb))
    elif pred_xgb is not None:
        final_pred = np.expm1(pred_xgb)
    elif pred_lgb is not None:
        final_pred = np.expm1(pred_lgb)
    else:
        return None

    return final_pred


# ================================================================
# Streamlit Frontend
# ================================================================
st.title("🏠 Ames Housing Price Prediction Dashboard")
st.write("Predict house prices using Machine Learning Models (XGBoost + LightGBM)")

st.sidebar.header("Model Load Status")
st.sidebar.write({
    "XGBoost Loaded": artifacts.get("xgb") is not None,
    "LightGBM Loaded": artifacts.get("lgb") is not None,
    "Scaler Loaded": artifacts.get("scaler") is not None,
})

mode = st.radio("Select Input Mode:", ["📄 CSV Upload", "🧮 Manual Entry"])

if mode == "📄 CSV Upload":
    file = st.file_uploader("Upload test CSV file", type=["csv"])
    if file:
        df_input = pd.read_csv(file)
        st.write("📋 Preview:", df_input.head())

        if st.button("🚀 Predict Prices"):
            preds = predict_price(df_input)
            st.success("Prediction Complete!")
            df_output = df_input.copy()
            df_output["PredictedPrice"] = preds
            st.write(df_output)
            st.download_button("⬇ Download Predictions",
                               df_output.to_csv(index=False),
                               file_name="predictions.csv")

elif mode == "🧮 Manual Entry":
    st.subheader("Enter House Details")

    GrLivArea = st.number_input("Above Ground Living Area (Sq ft):", min_value=400, max_value=5000)
    OverallQual = st.slider("Overall Material & Finish Quality", 1, 10, 5)

    df_input = pd.DataFrame({
        "GrLivArea": [GrLivArea],
        "OverallQual": [OverallQual]
    })

    if st.button("🔍 Predict"):
        pred = predict_price(df_input)
        if pred is not None:
            st.success(f"Estimated House Price: **${pred[0]:,.2f}**")
        else:
            st.error("Model not loaded properly!")

st.sidebar.title("ℹ️ About Project")
st.sidebar.write(
    "Interactive ML app that predicts Ames housing prices using ensemble Machine Learning models "
    "(XGBoost + LightGBM). Built and deployed by **Siddhant Srivastava** 🔥"
)
