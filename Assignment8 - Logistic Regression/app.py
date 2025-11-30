import streamlit as st
import pandas as pd
import numpy as np
import pickle



st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="💗",
    layout="centered"
)



st.markdown("""
<style>

:root {
    --primary-color: #0077b6;        /* Medical Blue */
    --secondary-color: #90e0ef;      /* Light Blue */
    --background-color: #f8f9fa;     /* Light Grey */
    --card-color: #ffffff;           /* White cards */
    --text-color: #003049;           /* Dark Navy text */
}

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: var(--background-color);
}

/* Card Style */
.card {
    background-color: var(--card-color);
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Titles */
h1, h2, h3 {
    color: var(--text-color) !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Input fields */
.stNumberInput label {
    color: var(--text-color) !important;
    font-weight: 600;
}
input {
    border-radius: 8px !important;
}

/* FANCY BUTTON */
div.stButton > button {
    background-color: var(--primary-color);
    color: white;
    padding: 12px 28px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: var(--secondary-color);
    color: black;
    transform: scale(1.05);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)


model = pickle.load(open("diabetes_logreg.pkl", "rb"))

st.markdown("<h1 style='text-align:center;'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#003049;'>A medical-grade risk assessment tool powered by Logistic Regression.</p>", unsafe_allow_html=True)


# User Inputs

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("Patient Information")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("BloodPressure", 0, 200, 70)
skin = st.number_input("SkinThickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 25)

st.markdown("</div>", unsafe_allow_html=True)


# Preprocessing for zero→median

def preprocess(value, median):
    return median if value == 0 else value


# Loading dataset to compute medians 

# Load dataset
df = pd.read_csv("diabetes.csv")

cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Safe replacement: column-wise loop (avoids recursion in pandas)
for col in cols_with_missing:
    df[col] = df[col].apply(lambda x: pd.NA if x == 0 else x)

# Fill missing values
medians = df[cols_with_missing].median()
for col in cols_with_missing:
    df[col] = df[col].fillna(medians[col])



# Replacing zero values with median

glucose = preprocess(glucose, medians["Glucose"])
bp = preprocess(bp, medians["BloodPressure"])
skin = preprocess(skin, medians["SkinThickness"])
insulin = preprocess(insulin, medians["Insulin"])
bmi = preprocess(bmi, medians["BMI"])


# Creating input array

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])


# Prediction

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🔴 Diabetes Detected! (Probability: {probability:.2f})")
    else:
        st.success(f"🟢 No Diabetes (Probability: {probability:.2f})")
