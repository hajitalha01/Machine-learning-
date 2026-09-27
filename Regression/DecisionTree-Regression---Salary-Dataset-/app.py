import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Salary Prediction (Decision Tree Regression)", page_icon="💼")

st.title("Salary Prediction App (Decision Tree Regression)")
st.caption("Feature: YearsExperience | Target: Salary")

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Sidebar: single prediction
st.sidebar.header("Single Prediction")
years_exp = st.sidebar.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1,
    value=5.0
)

if st.sidebar.button("Predict Salary"):
    pred = model.predict([[years_exp]])[0]
    st.sidebar.success(f"Predicted Salary: {pred:,.2f}")

# Main: batch predictions and metrics
st.subheader("Batch predictions and evaluation")
uploaded = st.file_uploader("Upload CSV with column 'YearsExperience' and optionally 'Salary'", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write("Preview:", data.head())

    # Check column
    if "YearsExperience" not in data.columns:
        st.error("CSV must contain 'YearsExperience' column.")
    else:
        # Predict
        preds = model.predict(data[["YearsExperience"]])
        data["PredictedSalary"] = preds
        st.write("Predictions:", data.head())

        # If ground-truth Salary is present, show metrics
        if "Salary" in data.columns:
            r2 = r2_score(data["Salary"], preds)
            mae = mean_absolute_error(data["Salary"], preds)
            rmse = mean_squared_error(data["Salary"], preds, squared=False)

            st.markdown("#### Evaluation metrics (on uploaded data)")
            cols = st.columns(3)
            cols[0].metric(label="R² Score", value=f"{r2:.4f}")
            cols[1].metric(label="MAE", value=f"{mae:,.2f}")
            cols[2].metric(label="RMSE", value=f"{rmse:,.2f}")

        # Download predictions
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

# Visuals section
st.subheader("Model tree visualization")
st.write("The training script has saved a static tree plot as 'decision_tree.png'. If present, it's displayed below.")

import os
if os.path.exists("decision_tree.png"):
    st.image("decision_tree.png", caption="Decision Tree Regressor", use_column_width=True)
else:
    st.info("decision_tree.png not found. Run the training script to generate it.")
