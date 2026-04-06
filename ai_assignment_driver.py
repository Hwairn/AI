import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

st.title("🩺 Diabetes Prediction System")
st.markdown("AI-powered prediction using Machine Learning")

# -------------------------
# LOAD FILES
# -------------------------
scaler = joblib.load("scaler.joblib")
gender_encoder = joblib.load("gender_encoder.joblib")
smoke_encoder = joblib.load("smoke_encoder.joblib")

accuracy = joblib.load("accuracy.joblib")
cm_dict = joblib.load("confusion_matrix.joblib")
report_dict = joblib.load("report.joblib")
roc_dict = joblib.load("roc.joblib")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ("ANN", "KNN", "SVM")
)

if model_choice == "ANN":
    model = joblib.load("ann_model.joblib")
elif model_choice == "KNN":
    model = joblib.load("knn_model.joblib")
else:
    model = joblib.load("svm_model.joblib")

# -------------------------
# TABS
# -------------------------
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Dashboard"])

# =========================
# 🔍 PREDICTION TAB
# =========================
with tab1:

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 1, 120, 30)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease History", [0, 1])

    with col2:
        smoking = st.selectbox(
            "Smoking History",
            ["never", "No Info", "current", "former", "ever", "not current"]
        )
        bmi = st.slider("BMI", 10.0, 60.0, 25.0)
        hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5)
        glucose = st.slider("Blood Glucose", 50, 300, 100)

    st.divider()

    if st.button("🚀 Predict Diabetes Risk"):

        gender_encoded = gender_encoder.transform([gender])[0]
        smoking_encoded = smoke_encoder.transform([smoking])[0]

        input_data = np.array([[gender_encoded, age, hypertension,
                                heart_disease, smoking_encoded,
                                bmi, hba1c, glucose]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        prob_no = probability[0][0]
        prob_yes = probability[0][1]

        st.subheader("📌 Prediction Result")

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk of Diabetes ({prob_yes*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({prob_no*100:.2f}%)")

        # METRICS
        colA, colB = st.columns(2)
        colA.metric("No Diabetes (%)", f"{prob_no*100:.2f}")
        colB.metric("Diabetes (%)", f"{prob_yes*100:.2f}")

        # PROGRESS BARS
        st.write("### 📊 Probability Breakdown")
        st.progress(int(prob_yes * 100))
        st.caption("Diabetes Probability")

        st.progress(int(prob_no * 100))
        st.caption("No Diabetes Probability")

# =========================
# 📊 DASHBOARD TAB
# =========================
with tab2:

    st.header("📊 Model Performance Dashboard")

    # -------------------------
    # Accuracy Section
    # -------------------------
    acc_df = pd.DataFrame({
        "Model": ["KNN", "SVM", "ANN"],
        "Accuracy (%)": [
            accuracy["KNN"] * 100,
            accuracy["SVM"] * 100,
            accuracy["ANN"] * 100
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Accuracy Table")
        st.dataframe(acc_df)

    with col2:
        st.subheader("📊 Accuracy Chart")
        st.bar_chart(acc_df.set_index("Model"))

    # Highlight best model
    best_model = max(accuracy, key=accuracy.get)
    st.success(f"🏆 Best Model: {best_model} ({accuracy[best_model]*100:.2f}%)")

    st.divider()

    # -------------------------
    # Detailed Analysis
    # -------------------------
    st.subheader("🔬 Detailed Model Analysis")

    selected_model = st.selectbox(
        "Choose Model for Analysis",
        ("KNN", "SVM", "ANN")
    )

# Confusion Matrix
    st.write("### Confusion Matrix")
    cm = cm_dict[selected_model]
    fig, ax = plt.subplots()

# Plot heatmap
    cax = ax.matshow(cm)

# Add color bar
    fig.colorbar(cax)
    
# Labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

# Show values inside boxes
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha='center', va='center')

    st.pyplot(fig)

    # Classification Report
    st.write("### Precision / Recall / F1-score")
    report = report_dict[selected_model]
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # ROC Curve
    st.write("### ROC Curve")

    roc = roc_dict[selected_model]

    fig, ax = plt.subplots()
    ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {roc['auc']:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{selected_model} ROC Curve")
    ax.legend()

    st.pyplot(fig)
