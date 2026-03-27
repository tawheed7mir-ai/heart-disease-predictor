import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ---------------- HEADER ----------------
st.markdown("<h1>❤️ Heart Disease AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Next-gen health risk prediction system</p>", unsafe_allow_html=True)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
show_chart = st.sidebar.toggle("Show Feature Chart", True)

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Towheed Qayoom")

# ---------------- INPUT ----------------
st.markdown("## 📋 Patient Input")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "ASY", "TA"])

with col2:
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    fasting_bs = st.selectbox("Fasting BS >120", [0, 1])

with col3:
    max_hr = st.number_input("Max HR", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# ---------------- PREDICT ----------------
if st.button("🚀 Predict"):

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

    # ---------------- RESULT ----------------
    st.markdown("## 📊 AI Dashboard")

    colA, colB = st.columns([2,1])

    # -------- GAUGE --------
    with colA:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Heart Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # -------- RESULT TEXT --------
    with colB:
        if prediction == 1:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")

        st.metric("Probability", f"{prob:.2f}")
        st.metric("Confidence", f"{prob*100:.1f}%")

    # -------- FEATURE CHART --------
    if show_chart:
        st.markdown("## 🧠 Feature Contribution (approx)")

        values = input_df.iloc[0].values
        feature_names = input_df.columns

        fig2 = go.Figure(
            go.Bar(
                x=feature_names,
                y=values
            )
        )
        fig2.update_layout(
            xaxis_title="Features",
            yaxis_title="Values"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<center>⚠️ Not medical advice. For educational use only.</center>",
    unsafe_allow_html=True
)
