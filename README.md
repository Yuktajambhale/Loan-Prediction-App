# 🏦 Loan Prediction Streamlit App

This is a Streamlit web application that predicts whether a loan application will be **Approved** or **Rejected** based on user inputs. It uses a pre-trained machine learning model trained on historical loan data.

## 🚀 Demo

🔗 [Live App on Streamlit](https://loan-prediction-app-f3hzbymvbri6w6shnpoz7a.streamlit.app/)

---

## 📂 Dataset

The dataset used for training is available here:

🔗 [Loan Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

You can also find the cleaned dataset used in this project as `clean_loan_data.csv` in the repo.

---

## 💻 Features

- User-friendly form to collect loan applicant info
- Predicts loan approval instantly
- Based on a trained Logistic Regression or Random Forest model
- Deployed on Streamlit Cloud

---

## 📁 Project Structure

project_loan/
│
├── app.py # Streamlit App code
├── train_model.py # Model training and saving code
├── loan_model.pkl # Trained ML model
├── scaler.pkl # Scaler for feature normalization
├── clean_loan_data.csv # Cleaned dataset
├── requirements.txt # Required Python packages

 Requirements
Python 3.8+

Streamlit

scikit-learn

pandas

joblib
