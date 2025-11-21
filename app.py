import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Streamlit Page Config
# =========================================================
st.set_page_config(
    page_title="Ames Housing Price Prediction",
    layout="wide"
)

# =========================================================
# Load Models & Artifacts
# =========================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts = {
        "xgb": None,
        "lgb": None,
        "scaler": None,
        "features": None,
        "train": None,
    }

    if os.path.exists("house_price_xgb_model.pkl"):
        artifacts["xgb"] = joblib.load("house_price_xgb_model.pkl")

    if os.path.exists("house_price_lgb_model.pkl"):
        artifacts["lgb"] = joblib.load("house_price_lgb_model.pkl")

    if os.path.exists("scaler.pkl"):
        artifacts["scaler"] = joblib.load("scaler.pkl")

    if os.path.exists("features.pkl"):
        try:
            artifacts["features"] = joblib.load("features.pkl")
        except Exception:
            artifacts["features"] = None

    if os.path.exists("train.csv"):
        try:
            artifacts["train"] = pd.read_csv("train.csv")
        except Exception:
            artifacts["train"] = None

    return artifacts


artifacts = load_artifacts()

# =========================================================
# Helper: Preprocessing
# (numeric alignment + scaling to match training)
# =========================================================
def preprocess_for_model(df_input: pd.DataFrame) -> np.ndarray:
    df = df_input.copy()

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Align to training feature columns
    feature_cols = artifacts.get("features")
    if feature_cols is not None:
        feature_cols = list(feature_cols)

        # Add missing columns
        missing_cols = [col for col in feature_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0

        # Drop extra columns
        extra_cols = [col for col in df.columns if col not in feature_cols]
        if extra_cols:
            df = df.drop(columns=extra_cols)

        # Reorder
        df = df[feature_cols]

    # Scale
    X = df.values
    scaler = artifacts.get("scaler")
    if scaler is not None:
        try:
            X = scaler.transform(df)
        except Exception:
            # Fallback to unscaled
            pass

    return X


# =========================================================
# Helper: Predict using Ensemble (XGB + LGB)
# =========================================================
def predict_prices(df_input: pd.DataFrame):
    X = preprocess_for_model(df_input)

    xgb = artifacts.get("xgb")
    lgb = artifacts.get("lgb")

    preds_xgb = None
    preds_lgb = None

    if xgb is not None:
        try:
            preds_xgb = xgb.predict(X)
        except Exception:
            preds_xgb = None

    if lgb is not None:
        try:
            preds_lgb = lgb.predict(X)
        except Exception:
            preds_lgb = None

    if preds_xgb is None and preds_lgb is None:
        return None, {"XGBoost": None, "LightGBM": None}

    # Convert from log1p-space back to original
    if preds_xgb is not None:
        preds_xgb_real = np.expm1(preds_xgb)
    else:
        preds_xgb_real = None

    if preds_lgb is not None:
        preds_lgb_real = np.expm1(preds_lgb)
    else:
        preds_lgb_real = None

    # Ensemble
    if preds_xgb_real is not None and preds_lgb_real is not None:
        ensemble = 0.6 * preds_xgb_real + 0.4 * preds_lgb_real
    elif preds_xgb_real is not None:
        ensemble = preds_xgb_real
    else:
        ensemble = preds_lgb_real

    model_wise = {
        "XGBoost": float(preds_xgb_real[0]) if preds_xgb_real is not None and len(preds_xgb_real) > 0 else None,
        "LightGBM": float(preds_lgb_real[0]) if preds_lgb_real is not None and len(preds_lgb_real) > 0 else None,
    }

    return ensemble, model_wise


# =========================================================
# Sidebar: Status & About
# =========================================================
st.sidebar.title("📦 Model Status")
st.sidebar.write({
    "XGBoost Loaded": artifacts["xgb"] is not None,
    "LightGBM Loaded": artifacts["lgb"] is not None,
    "Scaler Loaded": artifacts["scaler"] is not None,
    "Features Loaded": artifacts["features"] is not None,
    "Train Data Loaded": artifacts["train"] is not None,
})

st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About")
st.sidebar.write(
    "Interactive ML dashboard to predict **Ames housing prices** using an ensemble of "
    "XGBoost and LightGBM models. Includes quick prediction, batch CSV prediction, "
    "and model insights.\n\n"
    "**Built by Siddhant Srivastava**"
)
st.sidebar.markdown(
    "[GitHub Repo](https://github.com/SidGitCheck/ames-housing-price-dashboard)"
)

# =========================================================
# Main Layout
# =========================================================
st.title("🏠 Ames Housing Price Prediction — Interactive Dashboard")
st.markdown(
    "Use this tool to run quick what-if analyses, batch predictions on CSV files, "
    "and explore model behavior through visual insights."
)

tab_simple, tab_advanced, tab_insights = st.tabs(
    ["⚡ Quick Prediction", "📂 Advanced (CSV Upload)", "📊 Model Insights"]
)

# =========================================================
# Tab 1: Quick Prediction
# =========================================================
with tab_simple:
    st.subheader("⚡ Quick Prediction (Simple Mode)")

    st.info(
        "Default values below represent typical Ames housing characteristics. "
        "You can adjust them to explore different scenarios."
    )

    train_df = artifacts.get("train")

    def get_default(col, fallback):
        if train_df is not None and col in train_df.columns:
            try:
                return float(train_df[col].median())
            except Exception:
                return fallback
        return fallback

    col1, col2, col3 = st.columns(3)

    with col1:
        overall_qual = st.slider(
            "Overall Quality (1–10)",
            min_value=1,
            max_value=10,
            value=int(get_default("OverallQual", 5)),
        )
        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=300,
            max_value=6000,
            value=int(get_default("GrLivArea", 1500)),
            step=50,
        )
        garage_cars = st.slider(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=4,
            value=int(get_default("GarageCars", 2)),
        )

    with col2:
        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=3000,
            value=int(get_default("TotalBsmtSF", 800)),
            step=50,
        )
        full_bath = st.slider(
            "Full Bathrooms",
            min_value=0,
            max_value=4,
            value=int(get_default("FullBath", 2)),
        )
        half_bath = st.slider(
            "Half Bathrooms",
            min_value=0,
            max_value=3,
            value=int(get_default("HalfBath", 1)),
        )

    with col3:
        year_built = st.number_input(
            "Year Built",
            min_value=1870,
            max_value=2025,
            value=int(get_default("YearBuilt", 1990)),
            step=1,
        )
        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1000,
            max_value=50000,
            value=int(get_default("LotArea", 8000)),
            step=100,
        )
        bedroom_abv = st.slider(
            "Bedrooms Above Ground",
            min_value=1,
            max_value=8,
            value=int(get_default("BedroomAbvGr", 3)),
        )

    # Build input row (numeric-only, other features will be filled as 0 in preprocess)
    simple_input = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "YearBuilt": year_built,
        "LotArea": lot_area,
        "BedroomAbvGr": bedroom_abv,
    }])

    if st.button("🚀 Predict Price (Simple Mode)"):
        ensemble_pred, model_wise = predict_prices(simple_input)

        if ensemble_pred is None:
            st.error("Models are not loaded properly or input is invalid.")
        else:
            price = float(ensemble_pred[0])
            st.success(f"Estimated Sale Price: **${price:,.0f}**")

            with st.expander("🔍 Model-wise Predictions"):
                st.write(model_wise)


# =========================================================
# Tab 2: Advanced CSV Upload
# =========================================================
with tab_advanced:
    st.subheader("📂 Advanced Mode — Batch Predictions from CSV")

    st.markdown(
        "Upload a CSV file containing house features. The app will apply the same "
        "preprocessing pipeline and generate price predictions for each row."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("📋 Preview of uploaded data:")
        st.dataframe(df_uploaded.head())

        if st.button("🚀 Run Batch Predictions"):
            ensemble_pred, _ = predict_prices(df_uploaded)

            if ensemble_pred is None:
                st.error("Models could not generate predictions. Please check your input format.")
            else:
                df_results = df_uploaded.copy()
                df_results["Predicted_SalePrice"] = ensemble_pred
                st.success("Batch predictions completed.")
                st.dataframe(df_results.head())

                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Predictions as CSV",
                    data=csv_data,
                    file_name="ames_housing_predictions.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload a CSV file to run advanced batch predictions.")


# =========================================================
# Tab 3: Model Insights
# =========================================================
with tab_insights:
    st.subheader("📊 Model & Data Insights")

    train_df = artifacts.get("train")

    if train_df is not None:
        st.markdown("### 🔎 Sample of Training Data")
        st.dataframe(train_df.head())

        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

        # Correlation Heatmap
        if "SalePrice" in numeric_cols:
            st.markdown("### 🔥 Correlation Heatmap (Numeric Features)")
            corr = train_df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="viridis", ax=ax)
            st.pyplot(fig)

            # Distribution of SalePrice
            st.markdown("### 💰 Distribution of SalePrice")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(train_df["SalePrice"], kde=True, ax=ax2)
            ax2.set_xlabel("Sale Price")
            st.pyplot(fig2)

    else:
        st.info(
            "To see data-driven insights, ensure `train.csv` from the Ames Housing "
            "dataset is placed in the same directory as this app."
        )

    # Feature Importance (XGBoost)
    st.markdown("### 🧠 Top Features Driving Predictions (XGBoost)")

    xgb = artifacts.get("xgb")
    feature_cols = artifacts.get("features")

    if xgb is not None and feature_cols is not None:
        try:
            importances = xgb.feature_importances_
            if len(importances) == len(feature_cols):
                fi_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(15)

                fig3, ax3 = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    data=fi_df.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    ax=ax3
                )
                ax3.set_title("Top 15 Important Features (XGBoost)")
                st.pyplot(fig3)

                with st.expander("📋 View Feature Importance Table"):
                    st.dataframe(fi_df.reset_index(drop=True))
            else:
                st.warning("Feature importance length does not match feature columns. Skipping chart.")
        except Exception:
            st.warning("Could not compute feature importance from XGBoost model.")
    else:
        st.info("XGBoost model or feature columns not available for importance visualization.")
