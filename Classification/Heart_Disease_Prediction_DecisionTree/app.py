# -*- coding: utf-8 -*-
"""
Heart Disease Prediction App - Streamlit
"""

import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
# Make sure HeartDisease_Model.pkl is in the same folder as app.py
with open("HeartDisease_Model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below to predict the likelihood of heart disease.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.number_input("Fasting Blood Sugar (>120 mg/dl: 1, else 0)", min_value=0, max_value=1, value=0)
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -----------------------------
# Prepare input dataframe
# -----------------------------
input_dict = {
    "Age": [age],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "MaxHR": [max_hr],
    "Oldpeak": [oldpeak],
    "Sex": [sex],
    "ChestPainType": [chest_pain],
    "RestingECG": [resting_ecg],
    "ExerciseAngina": [exercise_angina],
    "ST_Slope": [st_slope]
}

input_df = pd.DataFrame(input_dict)

# -----------------------------
# One-hot encode categorical columns
# -----------------------------
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
input_df = pd.get_dummies(input_df, columns=categorical_cols)

# -----------------------------
# Define model columns manually (must match training)
# -----------------------------
model_columns = [
    'Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak',
    'Sex_F','ChestPainType_ATA','ChestPainType_ASY','ChestPainType_NAP',
    'RestingECG_LVH','RestingECG_ST','ExerciseAngina_Y',
    'ST_Slope_Flat','ST_Slope_Up'
]

# Add missing columns with 0
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[model_columns]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"Prediction: Heart Disease Likely ❤️ (Probability: {probability:.2f})")
    else:
        st.success(f"Prediction: Heart Disease Unlikely ✅ (Probability: {probability:.2f})")
