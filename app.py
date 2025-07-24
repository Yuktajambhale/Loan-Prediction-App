import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")



st.title("🏦 Loan Approval Prediction App")

# Sidebar Inputs
st.sidebar.header("Applicant Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.sidebar.number_input("Applicant Income", value=5000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", value=0)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", value=100)
loan_term = st.sidebar.number_input("Loan Term (in days)", value=360)
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])

# Prepare raw input
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Match encoding used in training
def encode_inputs(df):
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(int)
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    return df

input_encoded = encode_inputs(input_data)

# Scale numerical columns
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
