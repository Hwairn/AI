import streamlit as st
import joblib
import numpy as np

# Load ANN model
model = joblib.load("ann_model.joblib")

# Set title
st.title("Heart Disease Prediction")

# Example input data (replace with user input from Streamlit)
input_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Make prediction
prediction = model.predict(input_data)

# Show result on Streamlit page
if prediction[0] == 1:
    st.error("⚠️ Heart Disease Risk Detected")
else:
    st.success("✅ No Heart Disease Detected")
