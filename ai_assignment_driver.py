import streamlit as st
import joblib
import numpy as np

# Page title
st.title("Heart Disease Prediction System")

st.write("Testing with a predefined healthy input.")

# Load scaler and model
scaler = joblib.load("scaler.joblib")

# Choose the model to test
model_choice = st.selectbox(
    "Select Machine Learning Model",
    ("ANN", "KNN", "SVM")
)

# Load the corresponding model
if model_choice == "ANN":
    model = joblib.load("ann_model.joblib")
elif model_choice == "KNN":
    model = joblib.load("knn_model.joblib")
else:
    model = joblib.load("svm_model.joblib")

# -------------------------
# Hardcoded safe healthy input
# -------------------------
# Safe healthy input:
# age=35, sex=0, cp=1, trestbps=110, chol=180, fbs=0,
# restecg=0, thalach=190, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2
input_data = np.array([[35, 0, 1, 110, 180, 0, 0, 190, 0, 0.0, 2, 0, 2]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)

# Show result
st.write("Input data:", input_data)
if prediction[0] == 1:
    st.error("⚠️ Heart Disease Risk Detected")
else:
    st.success("✅ No Heart Disease Detected")
