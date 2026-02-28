import streamlit as st
import numpy as np
import joblib

model=joblib.load("diabetes_model.pkl")
scaler=joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

preg=st.number_input("Pregnancies")
glucose=st.number_input("Glucose")
bp=st.number_input("Blood Pressure")
skin=st.number_input("Skin Thickness")
insulin=st.number_input("Insulin")
bmi=st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
