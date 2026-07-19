#Setup and Libraries
# used to data handling
import pandas as pd
#streamlit is used to create the web app
import streamlit as st
#is used to load the model
import joblib

#Loading Model and Preprocessing Objects   
model = joblib.load('LR_ford_car.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

#Page Configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered")

#Title and Description
st.title("Ford Car Price Predictor")
st.write("Enter the car details below to predict its selling price.")

#Numerical Input Fields
year = st.number_input(
    "Manufacturing Year", 
    min_value=2000, 
    max_value=2026, 
    value=2020)
mileage = st.number_input(
    "Mileage", 
    min_value=0, 
    value=50000)
tax = st.number_input(
    "Road Tax (tax)", 
    min_value=0, 
    value=5000)
mpg = st.number_input(
    "MPG", 
    min_value=0, 
    value=30)
engine_size = st.number_input(
    "Engine Size", 
    min_value=0.0, 
    value=2.0, 
    step=0.1)

#Categorical Input using Dropdowns
Transmission= st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "semi-auto"]
)
Fuel_Type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "etc"])

#Text Input and Predict Button
Car_model = st.text_input("Car Model")
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size],
        "transmission": [Transmission],
        "fuelType": [Fuel_Type],
        "model": [Car_model]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=columns, fill_value=0)

    input_data[["year","mileage","tax","mpg","engineSize"]] = scaler.transform(
        input_data[["year","mileage","tax","mpg","engineSize"]]
    )

    prediction = model.predict(input_data)

    st.success(prediction[0])
