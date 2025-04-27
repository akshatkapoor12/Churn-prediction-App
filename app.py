import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Customer Churn Prediction App")

# Create form
gender = st.selectbox("Gender", ("Male", "Female"))
SeniorCitizen = st.selectbox("Senior Citizen", (0, 1))
Partner = st.selectbox("Has Partner?", ("Yes", "No"))
Dependents = st.selectbox("Has Dependents?", ("Yes", "No"))
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
PhoneService = st.selectbox("Phone Service", ("Yes", "No"))
InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
Contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
PaperlessBilling = st.selectbox("Paperless Billing", ("Yes", "No"))
PaymentMethod = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# Encoding manually based on previous label encoding
gender = 1 if gender == "Male" else 0
Partner = 1 if Partner == "Yes" else 0
Dependents = 1 if Dependents == "Yes" else 0
PhoneService = 1 if PhoneService == "Yes" else 0
InternetService = {"DSL":0, "Fiber optic":1, "No":2}[InternetService]
Contract = {"Month-to-month":0, "One year":1, "Two year":2}[Contract]
PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0
PaymentMethod = {"Electronic check":0, "Mailed check":1, "Bank transfer (automatic)":2, "Credit card (automatic)":3}[PaymentMethod]

# Predict button
if st.button('Predict Churn'):
    # Arrange features in same order
    features = np.array([[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                          InternetService, 0, 0, 0, 0, 0, 0, 0, 0, Contract,
                          PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]])
    
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn ❌")
    else:
        st.success("Customer is likely to Stay ✅")

