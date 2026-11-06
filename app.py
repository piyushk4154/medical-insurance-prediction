
import streamlit as st
import pandas as pd
import joblib

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load model
model = joblib.load("insurance_model.pkl")
columns = joblib.load("columns.pkl")

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details to predict insurance charges.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Prepare input
input_data = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": {"male": 1, "female": 0}[sex],
    "smoker": {"yes": 1, "no": 0}[smoker],
    "region": {
        "southwest": 0,
        "southeast": 1,
        "northwest": 2,
        "northeast": 3
    }[region]
}])

input_data = input_data[columns]

# Prediction
if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")
