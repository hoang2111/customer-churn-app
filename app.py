import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved artifacts
model = joblib.load('churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📞 Telco Customer Churn Predictor")
st.write("Fill in the customer profile below to calculate risk.")

# 2. Main Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial & Tenure")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    st.subheader("Demographics & Core Services")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

# 3. Hidden "Advanced Services" section to keep the UI clean
with st.expander("Add Digital & Streaming Services"):
    col3, col4 = st.columns(2)
    with col3:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with col4:
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# 4. Prediction Logic
if st.button("Calculate Churn Risk", use_container_width=True):
    # This dictionary must match your training column names exactly
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Process and Predict
    processed_data = preprocessor.transform(input_df)
    prob = model.predict_proba(processed_data)[0][1]

    # Visual result
    st.divider()
    if prob > 0.5:
        st.error(f"### High Risk: {prob:.1%} probability of churn")
        st.warning("Recommendation: Offer a long-term contract discount or loyalty bonus.")
    else:
        st.success(f"### Low Risk: {prob:.1%} probability of churn")
        st.info("Recommendation: Maintain current service level.")