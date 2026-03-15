import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Correct paths (matching your folder names)
MODEL_PATH = BASE_DIR / "Models" / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "Models" / "scaler.pkl"

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# App title
st.title("Diabetes Prediction System")

st.write("Enter the patient's medical details below:")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prediction
if st.button("Predict"):

    input_data = np.array([[pregnancies,
                            glucose,
                            blood_pressure,
                            skin_thickness,
                            insulin,
                            bmi,
                            dpf,
                            age]])

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("The patient is likely diabetic.")
    else:
        st.success("The patient is not diabetic.")