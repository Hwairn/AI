import streamlit as st
import joblib
import numpy as np

# Page title
st.title("Heart Disease Prediction System")

st.write("Enter patient information below to predict heart disease risk.")
scaler = joblib.load("scaler.joblib")
# Model selection
model_choice = st.selectbox(
    "Select Machine Learning Model",
    ("ANN", "KNN", "SVM")
)

# Load model based on user selection
if model_choice == "ANN":
    model = joblib.load("ann_model.joblib")
elif model_choice == "KNN":
    model = joblib.load("knn_model.joblib")
else:
    model = joblib.load("svm_model.joblib")

# Input fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False (0)", "True (1)"])
restecg = st.number_input("Resting ECG (0-2)", 0, 2)
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0)
slope = st.number_input("Slope (0-2)", 0, 2)
ca = st.number_input("Number of Major Vessels (0-3)", 0, 3)
thal = st.number_input("Thalassemia (1-3)", 1, 3)

# Convert categorical text to numeric
sex = 1 if sex == "Male (1)" else 0
fbs = 1 if fbs == "True (1)" else 0
exang = 1 if exang == "Yes (1)" else 0

# Predict button
if st.button("Predict Heart Disease Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Risk Detected")
    else:
        st.success("✅ No Heart Disease Detected")
