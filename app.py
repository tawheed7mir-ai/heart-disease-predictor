import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction App")

# User Inputs
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure")
cholesterol = st.number_input("Cholesterol")
fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate")
exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak")
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert to DataFrame
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

# Create base dataframe
input_df = pd.DataFrame([input_dict])

# Add dummy columns manually
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Handle categorical encoding
if "ChestPainType_" + chest_pain in input_df.columns:
    input_df["ChestPainType_" + chest_pain] = 1

if "RestingECG_" + rest_ecg in input_df.columns:
    input_df["RestingECG_" + rest_ecg] = 1

if "ST_Slope_" + st_slope in input_df.columns:
    input_df["ST_Slope_" + st_slope] = 1

# Reorder columns
input_df = input_df[columns]

# Scale
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")