import pandas as pd
import joblib
import streamlit as st

model = joblib.load('isFlaggedFraud_model.pkl')
encoder = joblib.load('isFlaggedFraud_encoder.pkl')

amount = st.number_input("Enter transaction amount:")
transaction_type = st.selectbox("Select transaction type", ["transfer", "payment", "withdraw", "deposit"])
location = st.selectbox("Select location", ["ZA", "UK", "CA", "AU", "IN"])

if st.button("Detect Fraud"):
    sample_data = pd.DataFrame({
        "amount": [amount],
        "transaction_type": [transaction_type],
        "location": [location]
    })

    converted = encoder.transform(sample_data)
    prediction = model.predict(converted)

    if prediction[0] == 1:
        st.warning("Potential fraud detected!")
    else:
        st.success("Transaction seems legitimate.")

