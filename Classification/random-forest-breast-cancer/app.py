import streamlit as st
import pandas as pd
import joblib

# ✅ FIX 1: Sahi path use karein (Colab wala path hatao)
model = joblib.load("model.pkl")  # Ya "model (2).pkl" agar rename nahi kiya

# Page config
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Breast Cancer Prediction App")
st.write("Predict whether a breast tumor is benign or malignant using a Random Forest model.")
st.markdown("---")

# All 30 features (same order as training data)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.sidebar.header("Input Features")
input_data = {}

# Create input sliders in sidebar for each feature
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        min_value=0.0,
        max_value=5000.0,
        value=0.0
    )

# Convert to DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Display input values
st.subheader("📊 User Input Features")
st.write(input_df)
st.markdown("---")

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Display prediction
    st.subheader("🧠 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ Malignant (Cancer Detected)")
    else:
        st.success("✅ Benign (No Cancer Detected)")
    
    # Display prediction probability
    st.subheader("📈 Prediction Probability")
    st.write(f"Benign     : {probability[0][0]:.2%}")
    st.write(f"Malignant  : {probability[0][1]:.2%}")

st.caption("Developed using Streamlit & Random Forest ML Model")
