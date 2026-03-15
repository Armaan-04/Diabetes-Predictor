import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Title
st.title("Diabetes Prediction System")

st.write("Enter the patient's medical details below:")

# User inputs
pregnancies = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prediction button
if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display result
    if prediction[0] == 1:
        st.error("The patient is likely diabetic.")
    else:
        st.success("The patient is not diabetic.")