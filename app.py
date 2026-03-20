import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved artifacts
model = joblib.load('/Users/macbook/Downloads/churn_model.pkl')
preprocessor = joblib.load('/Users/macbook/Downloads/preprocessor.pkl')

st.title("📞 Telco Customer Churn Predictor")
st.write("Enter customer details below to see the likelihood of them leaving.")

# 2. Create the User Interface (Inputs)
# Match these to your original dataframe columns
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 3. Prediction Logic
if st.button("Predict Probability"):
    # Create a dataframe from input (Must match raw feature names)
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        # Add other features used in your model here...
    }])

    # Process the data using your saved preprocessor
    processed_data = preprocessor.transform(input_df)

    # Get prediction and probability
    prob = model.predict_proba(processed_data)[0][1]

    if prob > 0.5:
        st.error(f"High Risk: {prob:.1%} chance of churn.")
    else:
        st.success(f"Low Risk: {prob:.1%} chance of churn.")