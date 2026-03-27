import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide"
)

# ------------------ LOAD ------------------
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ About")
st.sidebar.info(
    "This AI model predicts the likelihood of heart disease based on patient data.\n\n"
    "Built using Machine Learning (KNN)."
)

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Tawheed Mir")

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align: center;'>❤️ Heart Disease Prediction AI</h1>
    <p style='text-align: center; font-size:18px;'>
    Smart health risk detection powered by Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------ INPUT SECTION ------------------
st.markdown("## 📋 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])

with col2:
    resting_bp = st.number_input("Resting Blood Pressure", value=120)
    cholesterol = st.number_input("Cholesterol", value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col3:
    max_hr = st.number_input("Max Heart Rate", value=150)
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", value=1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# ------------------ PREDICT BUTTON ------------------
if st.button("🚀 Run Prediction"):

    input_dict = {
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "ExerciseAngina": 1 if exercise_angina == "Yes" else 0
    }

    input_df = pd.DataFrame([input_dict])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    if "ChestPainType_" + chest_pain in input_df.columns:
        input_df["ChestPainType_" + chest_pain] = 1

    if "ST_Slope_" + st_slope in input_df.columns:
        input_df["ST_Slope_" + st_slope] = 1

    input_df = input_df[columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.divider()

    # ------------------ RESULT ------------------
    st.markdown("## 📊 Prediction Dashboard")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Risk Score", f"{prob:.2f}")

    with colB:
        st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")

    with colC:
        st.metric("Confidence", f"{prob*100:.1f}%")

    # Progress bar
    st.progress(int(prob * 100))

    # Color result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# ------------------ FOOTER ------------------
st.divider()

st.markdown(
    """
    <center>
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and should not be used as medical advice.
    </center>
    """,
    unsafe_allow_html=True
)