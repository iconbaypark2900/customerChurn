import streamlit as st
import requests

st.title("Customer Churn Prediction App")

# User input form
st.header("Enter Customer Data")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
monthly_spend = st.number_input("Monthly Spend ($)", min_value=0.0, max_value=1000.0, value=100.0, step=0.01)
contract_type = st.selectbox("Contract Type", ["monthly", "yearly", "two_year"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)

# Make a prediction
if st.button("Predict Churn"):
    data = {
        "age": age,
        "monthly_spend": monthly_spend,
        "contract_type": contract_type,
        "tenure": tenure
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            churn_prediction = "Yes" if result["churn"] else "No"
            st.success(f"Churn Prediction: {churn_prediction}")
            
            # Additional information
            st.info("Prediction Details:")
            st.json(data)
        else:
            st.error(f"Error: Unable to get prediction. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Unable to connect to the backend server. {str(e)}")

# Add a section for model training
st.header("Model Management")
if st.button("Retrain Model"):
    try:
        response = requests.post("http://localhost:8000/train")
        if response.status_code == 200:
            st.success("Model retrained successfully!")
        else:
            st.error(f"Error: Unable to retrain model. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Unable to connect to the backend server. {str(e)}")

# Add a health check section
st.header("Backend Health")
if st.button("Check Backend Health"):
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            st.success("Backend is healthy!")
        else:
            st.error(f"Error: Backend health check failed. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Unable to connect to the backend server. {str(e)}")